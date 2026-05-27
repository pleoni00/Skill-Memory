"""
Assistente minimale con memoria DAG — client A2A.
-------------------------------------
Variabili d'ambiente:
  LLM_BASE_URL   — base URL LLM    (default: https://api.x.ai/v1)
  LLM_MODEL      — modello         (default: grok-3-mini)
  LLM_API_KEY    — api key         (obbligatorio)
  A2A_SERVER_URL — URL server A2A  (default: http://localhost:8000)

Avvio:
  python main.py
"""

import os
import json
import asyncio

from dotenv import load_dotenv
from openai import OpenAI
from a2a.client import create_client
from a2a.helpers import new_text_message
from a2a.types import Role

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL",   "https://api.x.ai/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "grok-3-mini")
LLM_API_KEY    = os.environ.get("LLM_API_KEY",    "dummy")
A2A_SERVER_URL = os.environ.get("A2A_SERVER_URL", "http://localhost:8000")

SYSTEM_PROMPT = """
Sei un AI Systems Architect specializzato in:
...
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
        system += f"\n\nContesto dalla memoria:\n{context_text}"

    response = llm.chat.completions.create(
        model    = LLM_MODEL,
        messages = [{"role": "system", "content": system}] + history,
    )
    return response.choices[0].message.content


# ── A2A helpers ───────────────────────────────────────────────────────────────

async def a2a_search(client, history: list[dict], top_k: int = 5) -> list[dict]:
    """Chiama la skill retrieval sul server A2A."""
    payload = json.dumps({
        "skill":  "retrieval",
        "turns":  history[-6:],
        "top_k":  top_k,
    })

    message = new_text_message(payload, role=Role.ROLE_USER)
    response_text = ""

    async for chunk in client.send_message(message):
        if chunk.HasField("artifact_update"):
            for part in chunk.artifact_update.artifact.parts:
                if part.HasField("text"):
                    response_text += part.text

    try:
        return json.loads(response_text).get("nodes", [])
    except (json.JSONDecodeError, ValueError):
        return []


async def a2a_store(client, history: list[dict]) -> str:
    """Chiama la skill ingestion sul server A2A."""
    payload = json.dumps({
        "skill": "ingestion",
        "turns": history,
    })

    message = new_text_message(payload, role=Role.ROLE_USER)
    response_text = ""

    async for chunk in client.send_message(message):
        if chunk.HasField("artifact_update"):
            for part in chunk.artifact_update.artifact.parts:
                if part.HasField("text"):
                    response_text += part.text

    return response_text


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main():
    client = await create_client(A2A_SERVER_URL)
    print("Assistente pronto. Scrivi 'exit' per uscire.\n")

    store   = True
    history: list[dict] = []

    while True:
        try:
            user_input = input("Tu: ").strip()
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
        print(f"\nAssistente: {reply}\n")

        history.append({"role": "assistant", "content": reply})

        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

    if history and store:
        print("\nSalvataggio conversazione in memoria...")
        log = await a2a_store(client, history)
        print(log)


if __name__ == "__main__":
    asyncio.run(main())