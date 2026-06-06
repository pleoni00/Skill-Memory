import json

from core.entities import Chunk, Conversation
from core.interfaces import Extractor, EmbeddingService
from agents.llm_client import LLMClient
from agents.prompts import EXTRACTOR_PROMPT


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
            max_tokens = 2000,
        )

        try:
            items = json.loads(raw.strip())
        except json.JSONDecodeError:
            return []

        feedback_items = items.get("behavior_feedback", [])
        if not feedback_items:
            return []

        summaries = [item["description"] for item in feedback_items if item.get("description")]
        if not summaries:
            return []

        embeddings = self._embed.embed(summaries)

        chunks: list[Chunk] = []
        for item, emb in zip(feedback_items, embeddings):
            description = item.get("description", "")
            if not description:
                continue
            signals = item.get("signals", [])
            topic = item.get("signal_type", "behavior_feedback")
            if signals:
                topic = f"{topic}: {signals[0][:48]}"

            chunks.append(
                Chunk(
                    text                   = description,
                    topic                  = topic,
                    embedding              = emb,
                    source_conversation_id = conversation.id,
                )
            )

        return chunks
