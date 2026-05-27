from core.interfaces import LLMClient

class OpenAICompatibleLLMClient(LLMClient):
    """
    Funziona con qualsiasi server compatibile OpenAI:
    Grok (api.x.ai), Ollama, LM Studio, vLLM, ecc.
    """
 
    def __init__(self, model: str, base_url: str, api_key: str = "dummy"):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model  = model
 
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        from openai import RateLimitError
        import time
 
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model      = self._model,
                    max_tokens = max_tokens,
                    messages   = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ]
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
 
        raise RuntimeError("Rate limit superato dopo 3 tentativi")
 