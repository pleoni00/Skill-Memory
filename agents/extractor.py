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

        if not items:
            return []
        
        knowledge_chunks = items.get("knowledge", [])
        summaries      = [item["summary"] for item in knowledge_chunks]
        embeddings = self._embed.embed(summaries)

        return [
            Chunk(
                text                   = item["summary"],
                topic                  = item["topic"],
                embedding              = emb,
                source_conversation_id = conversation.id,
            )
            for item, emb in zip(knowledge_chunks, embeddings)
        ]