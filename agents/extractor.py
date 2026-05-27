import json

from core.entities import Chunk, Conversation
from core.interfaces import Extractor, EmbeddingService
from agents.llm_client import LLMClient

EXTRACTOR_PROMPT = """
You are an expert conversation intelligence extractor.

You receive a conversation between a user and an assistant.

Your task is to extract ONLY USER INFORMATION and organize it into TWO distinct layers:

------------------------------------------------------------
1. KNOWLEDGE LAYER (what the user cares about)
------------------------------------------------------------
Extract:
- macro-topics
- goals
- constraints
- domain-specific context
- stable preferences

This is long-term memory.

------------------------------------------------------------
2. BEHAVIOR FEEDBACK LAYER (how the assistant should behave)
------------------------------------------------------------
Extract ONLY signals about interaction quality, such as:
- what the user considers correct/incorrect
- complaints about retrieval, reasoning, verbosity, structure
- preferences on explanation style
- requests for behavior changes
- evaluation signals (explicit or implicit)

This is NOT knowledge. It is policy guidance.

------------------------------------------------------------

Rules:
1. Ignore assistant messages
2. Do NOT over-fragment topics
3. Merge related concepts into macro units
4. Do not infer beyond what is stated
5. Keep outputs compact but information-dense

------------------------------------------------------------

Output format (STRICT JSON):

{
  "knowledge": [
    {
      "topic": "...",
      "summary": "...",
      "signals": ["...", "..."]
    }
  ],
  "behavior_feedback": [
    {
      "signal_type": "retrieval_error | preference | complaint | evaluation | constraint",
      "description": "...",
      "severity": "low | medium | high"
    }
  ]
}

If nothing exists in a section, return empty arrays.
"""

class LLMExtractor(Extractor):

    def __init__(self, embedding_service: EmbeddingService, llm: LLMClient):
        self._embed = embedding_service
        self._llm   = llm

    def extract(self, conversation: Conversation) -> list[Chunk]:
        convo_text = "\n".join(
            f"{t.role.upper()}: {t.content}"
            for t in conversation.turns
        )

        raw = self._llm.complete(
            system     = EXTRACTOR_PROMPT,
            user       = convo_text,
            max_tokens = 15000,
        )

        try:
            items = json.loads(raw.strip())
        except json.JSONDecodeError:
            return []

        if not items:
            return []

        texts      = [item["text"] for item in items]
        embeddings = self._embed.embed(texts)

        return [
            Chunk(
                text                   = item["text"],
                topic                  = item["topic"],
                embedding              = emb,
                source_conversation_id = conversation.id,
            )
            for item, emb in zip(items, embeddings)
        ]