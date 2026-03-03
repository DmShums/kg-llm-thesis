from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel
import requests
from utils.constants import BinaryOutputFormat, BinaryOutputFormatWithReasoning, LLMCallOutput, TokensUsage


class QwenServer:    
    def __init__(self, base_url: str = "http://localhost:8000", **kwargs: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_context: list[dict[str, str]] = []
        self.max_new_tokens = kwargs.get("max_new_tokens", 50)
        self.temperature = kwargs.get("temperature", 0.7)
        self.response_format = kwargs.get("response_format", BinaryOutputFormat)
        self.top_p = kwargs.get("top_p", 1.0)

    def add_system_context(self, message: str) -> None:
        if len(self.chat_context) == 0 or self.chat_context[0]["role"] != "system":
            self.chat_context.insert(0, {"role": "system", "content": message})
        else:
            self.chat_context[0] = {"role": "system", "content": message}

    def add_context(self, message: str, role: str) -> None:
        self.chat_context.append({"role": role, "content": message})

    def clear_context(self) -> None:
        self.chat_context = []

    def ask_sync_question(self, message: str, model: str = "Qwen/Qwen3-1.7B") -> LLMCallOutput:
        try:
            messages = [*self.chat_context, {"role": "user", "content": message}]
            payload = {
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }

            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=1000,
            )
            response.raise_for_status()
            text_response = response.json().get("response", "").strip()

            parsed = None
            lower_text = text_response.lower()

            if lower_text in ["true", "yes", "1"]:
                parsed = BinaryOutputFormat(answer=True)
            elif lower_text in ["false", "no", "0"]:
                parsed = BinaryOutputFormat(answer=False)
            elif "reasoning:" in lower_text:
                reasoning, _, ans_text = lower_text.partition("answer:")
                parsed = BinaryOutputFormatWithReasoning(
                    reasoning=reasoning.replace("reasoning:", "").strip(),
                    answer=ans_text.strip().lower() in ["true", "yes", "1"],
                )

            usage = TokensUsage(input_tokens=None, output_tokens=None)
            logprobs = []

            return LLMCallOutput(
                message=text_response,
                usage=usage,
                logprobs=logprobs,
                parsed=parsed,
            )

        except Exception as e:
            raise RuntimeError(f"Request to Qwen server failed: {e}")