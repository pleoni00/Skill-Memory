import anthropic
from core.interfaces import EmbeddingService

MODEL = "text-embedding-3-small"


class AnthropicEmbeddingService(EmbeddingService):
    """
    Usa il client Anthropic per generare embedding.
    Nota: Anthropic non espone ancora un endpoint embedding nativo —
    usiamo OpenAI text-embedding-3-small come service esterno,
    wrappato in questa interfaccia per poterlo swappare facilmente.
    """

    def __init__(self, api_key: str):
        # Per ora usiamo openai direttamente dato che Anthropic
        # non ha ancora un endpoint embedding pubblico.
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._backend = "openai"
        except ImportError:
            raise RuntimeError(
                "openai package required for embeddings: pip install openai"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]


class LocalEmbeddingService(EmbeddingService):
    """
    Fallback senza API key: embedding con sentence-transformers locale.
    Dimensione output: 384 (all-MiniLM-L6-v2).
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required: pip install sentence-transformers"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
class OpenAICompatibleEmbeddingService(EmbeddingService):
    """
    Funziona con qualsiasi server che espone /v1/embeddings compatibile OpenAI:
    Ollama, LM Studio, llama.cpp, vLLM, ecc.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model  = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
