from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from openai import OpenAI

from metis_bot import MetisBotClient


# ------------------ model spec ------------------

@dataclass(frozen=True)
class ModelSpec:
    provider: str  # openai | anthropic | grok | deepseek | gemini | metis_bot
    model: Optional[str] = None  # model name (unused for bot)
    base_url: Optional[str] = None
    bot_id: Optional[str] = None  # ONLY for metis_bot


# ------------------ unified wrapper ------------------

class MultiLLM:
    def __init__(self, metis_api_key: str, metis_endpoint: str = "https://api.metisai.ir"):
        self.api_key = metis_api_key
        self.endpoint = metis_endpoint.rstrip("/")

        self.base_urls = {
            "openai": f"{self.endpoint}/openai/v1",
            "anthropic": f"{self.endpoint}/api/v1/wrapper/anthropic",
            "grok": f"{self.endpoint}/api/v1/wrapper/grok",
            "deepseek": f"{self.endpoint}/api/v1/wrapper/deepseek",
        }

        self._clients: Dict[str, OpenAI] = {}

        self.metis_bot = MetisBotClient(self.api_key)

    # ------------------ models ------------------

    def register_models(self) -> Dict[str, ModelSpec]:
        return {

            # ---------- Anthropic ----------
            "claude_opus": ModelSpec("anthropic", "claude-4-opus", self.base_urls["anthropic"]),
            "claude_sonnet": ModelSpec("anthropic", "claude-4-sonnet", self.base_urls["anthropic"]),
            "claude_haiku": ModelSpec("anthropic", "claude-3-haiku", self.base_urls["anthropic"]),

            # ---------- Grok ----------
            "grok_fast": ModelSpec("grok", "grok-4-fast", self.base_urls["grok"]),

            # ---------- DeepSeek ----------
            "deepseek_chat": ModelSpec("deepseek", "deepseek-chat", self.base_urls["deepseek"]),
            "deepseek_reasoner": ModelSpec("deepseek", "deepseek-reasoner", self.base_urls["deepseek"]),

            # ---------- Gemini ----------
            "gemini_pro": ModelSpec(provider="metis_bot",
                                    bot_id="0d167edd-f0b8-4d5e-84bf-36a005fae141"),
            "gemini_flash": ModelSpec(provider="metis_bot", bot_id="78a3f58d-235f-4ede-b033-178d7e01b291"),

            # ---------- OpenAI ----------
            "gpt4o": ModelSpec("openai", "gpt-4o", self.base_urls["openai"]),
            "gpt4o_mini": ModelSpec("openai", "gpt-4o-mini", self.base_urls["openai"]),
            "gpt41": ModelSpec("openai", "gpt-4.1", self.base_urls["openai"]),
            "gpt41_mini": ModelSpec("openai", "gpt-4.1-mini", self.base_urls["openai"]),

            "gpt5.2": ModelSpec(
                provider="metis_bot",
                bot_id="8b082625-285f-4340-a05b-b912cfb93ba0",
            ),
        }

    # ------------------ public API ------------------

    def chat(
            self,
            spec: ModelSpec,
            user_text: str,
            system_text: str = "You are a helpful assistant.",
            max_tokens: int = 512,
    ) -> str:

        if spec.provider == "metis_bot":
            resp = self.metis_bot.send_message(
                bot_id=spec.bot_id,
                prompt=user_text,
            )
            # Metis bot responses are structured
            return resp["content"]

        return self._chat_openai_compatible(
            base_url=spec.base_url,
            model=spec.model,
            user_text=user_text,
            system_text=system_text,
            max_tokens=max_tokens,
        )

    # ------------------ internals ------------------

    def _get_client(self, base_url: str) -> OpenAI:
        if base_url not in self._clients:
            self._clients[base_url] = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
            )
        return self._clients[base_url]

    def _chat_openai_compatible(
            self,
            base_url: str,
            model: str,
            user_text: str,
            system_text: str,
            max_tokens: int,
    ) -> str:

        client = self._get_client(base_url)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
        )

        return resp.choices[0].message.content


# ------------------ usage ------------------

if __name__ == "__main__":
    METIS_API_KEY = "tpsg-SqeTeqoHXs8MHgqdQ7ts0GdE5lxzTSe"

    llm = MultiLLM(METIS_API_KEY)
    models = llm.register_models()

    print(llm.chat(models["gpt4o"], "Hello world"))
    print(llm.chat(models["gpt5.2"], "Explain Gemini like I am a kid."))
    print(llm.chat(models["grok_fast"], "What is Grok?"))
    print(llm.chat(models["deepseek_chat"], "Explain transformers in 2 lines."))
    print(llm.chat(models["claude_opus"], "Explain Gemini like I am a kid."))
    print(llm.chat(models["gemini_pro"], "Explain Gemini like I am a kid."))
    print(llm.chat(models["gemini_flash"], "Explain Gemini like I am a kid."))
