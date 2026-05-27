"""
Assistente minimale con memoria DAG.
-------------------------------------
Variabili d'ambiente:
  LLM_BASE_URL   — base URL LLM    (default: https://api.x.ai/v1)
  LLM_MODEL      — modello         (default: grok-3-mini)
  LLM_API_KEY    — api key         (obbligatorio)
  MCP_SERVER_CMD — comando per avviare il server MCP
                   (default: python mcp/server.py)

Avvio:
  python main.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL",   "https://api.x.ai/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "grok-3-mini")
LLM_API_KEY    = os.environ.get("LLM_API_KEY",    "dummy")
MCP_SERVER_CMD = os.environ.get("MCP_SERVER_CMD", "python mcp/server.py").split()

SYSTEM_PROMPT = """
Sei un AI Systems Architect specializzato in:

- DAG memory systems
- hybrid vector + graph retrieval
- LLM orchestration
- agent memory design
- retrieval debugging

Obiettivo:
Aiutare l'utente a costruire, debug e ottimizzare sistemi di memoria per agenti AI.

Regole:
1. Non essere un assistente generico
2. Ogni risposta deve avere implicazioni architetturali
3. Se possibile usa tool per ispezionare il sistema
4. Evidenzia problemi strutturali (non solo sintattici)
5. Suggerisci miglioramenti concreti e implementabili
6. Pensa sempre in termini di:
   - retrieval
   - graph structure
   - embedding behavior
   - LLM decision policy

Quando rilevi problemi, classificali in:
- STRUCTURE ISSUE
- RETRIEVAL ISSUE
- MEMORY ISSUE
- SCALING ISSUE

Rispondi in modo conciso ma tecnico.
"""

MAX_HISTORY_TURNS = 10   # turni mantenuti in memoria per la sessione

# ── LLM client ────────────────────────────────────────────────────────────────

llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def chat(history: list[dict], context_nodes: list[dict]) -> str:
    """Chiama l'LLM con history + contesto DAG iniettato nel system prompt."""
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


# ── MCP helpers ───────────────────────────────────────────────────────────────

async def mcp_search(session: ClientSession, history: list[dict], top_k: int = 5) -> list[dict]:
    """Chiama il tool search passando gli ultimi N turni."""
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-6:]
    ]
    result = await session.call_tool("search", {"turns": turns, "top_k": top_k})
    try:
        data = json.loads(result.content[0].text)
        return data.get("nodes", [])
    except (json.JSONDecodeError, IndexError):
        return []


async def mcp_store(session: ClientSession, history: list[dict]) -> str:
    """Chiama store_conversation a fine sessione."""
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in history
    ]
    result = await session.call_tool("store_conversation", {"turns": turns})
    try:
        return result.content[0].text
    except IndexError:
        return ""


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main():
    server_params = StdioServerParameters(
        command = MCP_SERVER_CMD[0],
        args    = MCP_SERVER_CMD[1:],
        env     = os.environ.copy(),
    )

    store = True

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Assistente pronto. Scrivi 'exit' per uscire.\n")

            history: list[dict] = []

            while True:
                try:
                    user_input = input("Tu: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue

                if user_input.startswith("/config "):
                    context = user_input[8:].strip()
                    result  = await session.call_tool("configure", {"context": context})
                    print(result.content[0].text)
                    continue

                if user_input[:4] in ("exit", "quit", "esci"):
                    if user_input.endswith("no_store"):
                        store = False
                    break

                # Aggiunge il messaggio utente alla history
                history.append({"role": "user", "content": user_input})

                # Cerca contesto nel DAG
                nodes = await mcp_search(session, history)

                # Risponde con contesto
                reply = chat(history, nodes)
                print(f"\nAssistente: {reply}\n")

                # Aggiunge risposta alla history
                history.append({"role": "assistant", "content": reply})

                # Mantieni la history entro il limite
                if len(history) > MAX_HISTORY_TURNS * 2:
                    history = history[-(MAX_HISTORY_TURNS * 2):]

            # Fine sessione: salva in memoria
            if history and store:
                print("\nSalvataggio conversazione in memoria...")
                log = await mcp_store(session, history)
                print(log)


if __name__ == "__main__":
    asyncio.run(main())