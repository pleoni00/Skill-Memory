"""
Minimal assistant with DAG memory — A2A client.
-------------------------------------
Environment variables:
  LLM_BASE_URL   — LLM base URL    (default: https://api.x.ai/v1)
  LLM_MODEL      — model           (default: grok-3-mini)
  LLM_API_KEY    — API key         (required)
  A2A_SERVER_URL — A2A server URL  (default: http://localhost:8000)

Run:
  python main.py
"""

import os
import json
import asyncio
import httpx

from dotenv import load_dotenv
from openai import OpenAI
from a2a.client import create_client, ClientConfig
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, SendMessageConfiguration

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL",   "https://api.x.ai/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "grok-3-mini")
LLM_API_KEY    = os.environ.get("LLM_API_KEY",    "dummy")
A2A_SERVER_URL = os.environ.get("A2A_SERVER_URL", "http://localhost:8000")

SYSTEM_PROMPT = """
You are K8sRCAAgent, a specialized assistant for root cause analysis (RCA) of incidents
on Kubernetes clusters. Your sole focus is diagnosing why a failure occurred, identifying
the contributing factors, and recommending remediation and prevention steps.
 
You do NOT perform capacity planning, cost optimization, or general DevOps consulting
unless it directly explains a failure you are investigating.
 
─────────────────────────────────────────────────────────────────────────
ANALYTICAL FRAMEWORK
─────────────────────────────────────────────────────────────────────────
For every incident request, structure your response as follows:
 
  ## Incident Summary
  One-sentence headline: namespace / workload / failure type / impact / time.
 
  ## Signal Decomposition
  List the observable signals ranked by diagnostic weight (most informative first):
  - Signal name (e.g., OOMKilled, CrashLoopBackOff, Pending, Evicted)
  - Source (metrics, logs, events, alerts)
  - What it rules in / rules out
 
  ## Hypothesis Ranking
  List hypotheses from most to least likely. For each:
  - Hypothesis label
  - Supporting evidence
  - Falsifying condition (what would prove it wrong)
 
  ## Root Cause Assessment
  Distinguish:
  - Proximate cause (immediate trigger)
  - Contributing factors (conditions that made it possible)
  - Systemic risk (why the cluster had no protection against this)
 
  ## Recommended Remediation
  Immediate (< 1h), Short-term (< 1 week), Long-term (< 1 quarter).
  Flag if a post-mortem is warranted (default threshold: customer-facing impact > 5 min).
 
  ## Data Gaps
  Exact list of logs, metrics, or events needed to confirm or reject top hypotheses.
 
─────────────────────────────────────────────────────────────────────────
CONSTRAINTS
─────────────────────────────────────────────────────────────────────────
- Always distinguish OOMKill (kernel-level) from OOMKilled (k8s limit enforcement).
- Never assume a deploy caused an incident without evidence. Correlation in time is
  not causation.
- When data is insufficient for hypothesis ranking, return a structured data request
  listing exactly which kubectl commands or observability queries to run.
- Maintain technical neutrality: do not soften findings to protect a team or vendor.
- Default to the simplest explanation consistent with the evidence (Occam's razor).
"""

MAX_HISTORY_TURNS = 10

# ── LLM client ────────────────────────────────────────────────────────────────

llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def chat(history: list[dict], context_nodes: list[dict]) -> str:
    system = SYSTEM_PROMPT
    if context_nodes:
        context_text = "\n\n".join(
            f"[{n['title']}]\n{n['content']}"
            for n in context_nodes
        )
        system += f"\n\nContext from memory:\n{context_text}"

    response = llm.chat.completions.create(
        model    = LLM_MODEL,
        messages = [{"role": "system", "content": system}] + history,
    )
    return response.choices[0].message.content


# ── A2A helpers ───────────────────────────────────────────────────────────────

async def a2a_search(client, history: list[dict], top_k: int = 5) -> list[dict]:
    """Call the retrieval skill on the A2A server."""
    payload = json.dumps({
        "skill":  "retrieval",
        "turns":  history[-6:],
        "top_k":  top_k,
    })

    message = new_text_message(payload, role=Role.ROLE_USER)
    response_text = ""

    request = SendMessageRequest(
        message=message,
        configuration=SendMessageConfiguration()
    )

    async for chunk in client.send_message(request):
        if chunk.HasField("artifact_update"):
            for part in chunk.artifact_update.artifact.parts:
                if part.HasField("text"):
                    response_text += part.text

    try:
        return json.loads(response_text).get("nodes", [])
    except (json.JSONDecodeError, ValueError):
        return []


async def a2a_store(client, history: list[dict]) -> str:
    """Call the ingestion skill on the A2A server."""
    payload = json.dumps({
        "skill": "ingestion",
        "turns": history,
    })

    message = new_text_message(payload, role=Role.ROLE_USER)
    response_text = ""

    request = SendMessageRequest(
        message=message,
        configuration=SendMessageConfiguration()
    )

    async for chunk in client.send_message(request):
        if chunk.HasField("artifact_update"):
            for part in chunk.artifact_update.artifact.parts:
                if part.HasField("text"):
                    response_text += part.text

    return response_text


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main():
    timeout = httpx.Timeout(12000)
    httpx_client = httpx.AsyncClient(timeout=timeout)
    config = ClientConfig(httpx_client=httpx_client)
    client = await create_client(A2A_SERVER_URL, client_config=config)
    print("Assistant ready. Type 'exit' to quit.\n")

    store   = True
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input[:4] in ("exit", "quit", "esci"):
            if user_input.endswith("no_store"):
                store = False
            break

        history.append({"role": "user", "content": user_input})

        nodes = await a2a_search(client, history)
        reply = chat(history, nodes)
        print(f"\nAssistant: {reply}\n")

        history.append({"role": "assistant", "content": reply})

        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

    if history and store:
        print("\nSaving conversation to memory...")
        log = await a2a_store(client, history)
        print(log)


if __name__ == "__main__":
    asyncio.run(main())