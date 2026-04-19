from __future__ import annotations

import json
import os
from typing import Any

import openai
from pydantic import BaseModel, RootModel

from utils.constants import BinaryOutputFormat, LLMCallOutput, TokensUsage


class OpenRouterServer:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs: dict[str, Any]) -> None:
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_key_info = api_key

        self.base_url = "https://openrouter.ai/api/v1"
        if base_url is not None:
            self.base_url = base_url

        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url, max_retries=2)
        self.chat_context = []

        self.response_format = kwargs.get("response_format", BinaryOutputFormat)

        self.top_p = 0.3
        self.temperature = 0
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.logprobs = True
        self.top_logprobs = 3
        self.params = kwargs

        self.routes_registry = {
            # "qwen/qwen3-vl-8b-instruct": "novita",
            # "deepseek/deepseek-chat-v3-0324": "deepinfra",
            # "mistralai/ministral-14b-2512": "mistral",
            # "mistralai/ministral-3b-2512": "mistral",
            # "mistralai/mistral-nemo": "deepinfra",
            # "google/gemma-3-27b-it": "deepinfra",
            # "google/gemma-3-12b-it": "deepinfra",
            # "meta-llama/llama-4-scout": "deepinfra",
            # "meta-llama/llama-4-maverick": "deepinfra",
            # "openai/gpt-oss-120b": "deepinfra",
            # "x-ai/grok-4.1-fast": "xai",
            # "qwen/qwen3-235b-a22b-2507": "deepinfra",
            # "openai/gpt-4o-mini": "openai",
            # "google/gemini-3-flash-preview": "google-ai-studio",
            # "google/gemini-2.5-pro": "google-vertex/global",
            # "mistralai/mistral-small-3.2-24b-instruct": "mistral",
            # "meta-llama/llama-3.1-70b-instruct": "parasail",
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_system_context(self, message: str) -> None:
        """Add or update the system context in the chat history.

        If a system message exists, it will be updated; otherwise, it will be added.
        """
        if len(self.chat_context) == 0 or self.chat_context[0]["role"] != "system":
            self.chat_context.insert(0, self.wrap_text_message(message, "system"))
        else:
            self.chat_context[0] = self.wrap_text_message(message, "system")

    def add_context(self, message: str, role: str) -> None:
        """Add a message to the chat context."""
        self.chat_context.append(self.wrap_text_message(message, role))

    def set_response_format(self, response_format: BaseModel | dict) -> None:
        """Set the response format for the server."""
        self.response_format = response_format

    def wrap_text_message(self, message: str, role: str) -> dict:
        """Wrap a text message with a role."""
        return {"role": role, "content": message}

    def ask_sync_question(self, message: str, model: str) -> LLMCallOutput:
        """
        Send a question to the LLM and return a validated LLMCallOutput.
        Handles variations in LLM output formats to ensure BinaryOutputFormat is valid.
        """
        try:
            # Build messages: chat history + current user message
            messages = [*self.chat_context, self.wrap_text_message(message, "user")]
            model = model.replace("|", "/")

            # Provider routing
            extra_body = {}
            if route := self.routes_registry.get(model):
                extra_body.setdefault("provider", {})
                extra_body["provider"].setdefault("only", [route])

            # Construct inference arguments (do NOT pass response_format here)
            inference_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "extra_body": extra_body,
            }

            # Optional reasoning parameters
            if self.params.get("reasoning_effort"):
                inference_kwargs["reasoning_effort"] = self.params.get("reasoning_effort")
            if self.params.get("thinking_budget"):
                inference_kwargs.setdefault("extra_body", {})
                inference_kwargs["extra_body"].setdefault("reasoning", {})
                inference_kwargs["extra_body"]["reasoning"]["max_tokens"] = int(self.params.get("thinking_budget"))

            # Call LLM (raw response)
            # NOTE: use the raw chat completion create / parse endpoint that returns the textual content
            response = self.client.beta.chat.completions.parse(**inference_kwargs)
            output_message = response.choices[0].message.content.strip()
            # print("LLM raw output:", repr(output_message))

            # -----------------------------
            # Robust parsing for BinaryOutputFormat
            # -----------------------------
            # Try these in order:
            # 1) exact words "true"/"false" (case-insensitive)
            # 2) simple JSON ({"answer": true, ...})
            # 3) find first {...} substring and try to parse
            # 4) fuzzy search for 'true' or 'false' tokens
            normalized = output_message.strip()
            # 1) bare booleans, possibly quoted
            tmp = normalized.strip().strip('"').strip("'").lower()
            if tmp == "true":
                parsed_output = self.response_format(answer=True)
            elif tmp == "false":
                parsed_output = self.response_format(answer=False)
            else:
                parsed_output = None
                # 2) try parse as JSON directly
                try:
                    parsed_output = self.response_format(**json.loads(normalized))
                except Exception:
                    # 3) try to extract a JSON object substring (the model sometimes wraps JSON inside text)
                    try:
                        start = normalized.index("{")
                        end = normalized.rindex("}") + 1
                        json_sub = normalized[start:end]
                        parsed_output = self.response_format(**json.loads(json_sub))
                    except Exception:
                        parsed_output = None

                # 4) fallback: simple token search for 'true'/'false' in text
                if parsed_output is None:
                    if " true" in normalized.lower() or normalized.lower().startswith("true"):
                        parsed_output = self.response_format(answer=True)
                    elif " false" in normalized.lower() or normalized.lower().startswith("false"):
                        parsed_output = self.response_format(answer=False)
                    else:
                        # last-resort: no match -> conservative false (or you can choose None)
                        parsed_output = self.response_format(answer=False)

            # -----------------------------
            # Track token usage
            # -----------------------------
            usage = TokensUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            try:
                logprobs = response.choices[0].logprobs.model_dump()["content"]
            except AttributeError:
                logprobs = []

            return LLMCallOutput(
                message=output_message,
                usage=usage,
                logprobs=logprobs,
                parsed=parsed_output,
            )

        except Exception as e:
            raise RuntimeError(f"ask_sync_question failed: {e}") from e
    
    def ask_chat(self, messages: list[dict], model: str) -> LLMCallOutput:
        """
        Send structured chat messages (few-shot / RAG).
        Messages must already contain roles.
        """
        try:
            model = model.replace("|", "/")

            # Provider routing
            extra_body = {}
            if route := self.routes_registry.get(model):
                extra_body.setdefault("provider", {})
                extra_body["provider"].setdefault("only", [route])

            inference_kwargs = {
                "model": model,
                "messages": messages,   # ← IMPORTANT: no wrapping
                "temperature": self.temperature,
                "top_p": self.top_p,
                "extra_body": extra_body,
            }

            if self.params.get("reasoning_effort"):
                inference_kwargs["reasoning_effort"] = self.params.get("reasoning_effort")

            if self.params.get("thinking_budget"):
                inference_kwargs.setdefault("extra_body", {})
                inference_kwargs["extra_body"].setdefault("reasoning", {})
                inference_kwargs["extra_body"]["reasoning"]["max_tokens"] = int(
                    self.params.get("thinking_budget")
                )

            response = self.client.beta.chat.completions.parse(**inference_kwargs)

            output_message = response.choices[0].message.content.strip()
            # print("LLM raw output:", repr(output_message))

            # ---- reuse SAME parsing logic ----
            import re
            import json

            normalized = output_message.strip()

            parsed_output = None

            # try JSON first (best case)
            try:
                parsed_output = self.response_format(**json.loads(normalized))
            except Exception:
                pass

            # try extracting JSON substring
            if parsed_output is None:
                try:
                    start = normalized.index("{")
                    end = normalized.rindex("}") + 1
                    parsed_output = self.response_format(
                        **json.loads(normalized[start:end])
                    )
                except Exception:
                    pass

            # detect true/false anywhere in the text
            if parsed_output is None:
                match = re.search(r"\b(true|false)\b", normalized.lower())
                if match:
                    parsed_output = self.response_format(
                        answer=(match.group(1) == "true")
                    )
                else:
                    parsed_output = self.response_format(answer=False)

            usage = TokensUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            try:
                logprobs = response.choices[0].logprobs.model_dump()["content"]
            except AttributeError:
                logprobs = []

            return LLMCallOutput(
                message=output_message,
                usage=usage,
                logprobs=logprobs,
                parsed=parsed_output,
            )

        except Exception as e:
            raise RuntimeError(f"ask_chat failed: {e}") from e


    def ask_structured_question(
        self,
        message: str,
        model: str,
        response_model: type[BaseModel],
    ) -> LLMCallOutput:

        try:
            self.clear_context()

            messages = [self.wrap_text_message(message, "user")]
            model = model.replace("|", "/")

            extra_body = {}
            if route := self.routes_registry.get(model):
                extra_body.setdefault("provider", {})
                extra_body["provider"].setdefault("only", [route])

            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=extra_body,
            )

            output_message = response.choices[0].message.content.strip()
            # print("LLM raw output:", repr(output_message))

            # -----------------------
            # STRICT JSON PARSING
            # -----------------------
            parsed_output = None

            try:
                data = json.loads(output_message)
            except Exception:
                # try extracting JSON block
                try:
                    start = output_message.index("{")
                    end = output_message.rindex("}") + 1
                    data = json.loads(output_message[start:end])
                except Exception as e:
                    raise RuntimeError(
                        "LLM did not return valid JSON.\n"
                        f"Output:\n{output_message}"
                    ) from e

            # Validate via Pydantic
            if issubclass(response_model, RootModel):
                parsed_output = response_model.model_validate(data)
            else:
                parsed_output = response_model(**data)

            usage = TokensUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            return LLMCallOutput(
                message=output_message,
                usage=usage,
                logprobs=[],
                parsed=parsed_output,
            )

        except Exception as e:
            raise RuntimeError(f"ask_structured_question failed: {e}") from e
    
    def ask_repair_ranking(
        self,
        prompt: str,
        model: str = "qwen/qwen3-vl-8b-instruct",
        repair_type: str = "all"
    ) -> LLMCallOutput:

        from utils.constants import RepairRankingOutput

        old_format = self.response_format   # <- save

        try:
            self.clear_context()

            self.set_response_format(RepairRankingOutput)

            if repair_type == "all":
                # self.add_system_context(
                #     "Return ONLY valid JSON matching the required schema."
                # )
                self.add_system_context("""
You MUST respond ONLY with valid JSON.
Do NOT explain.
Do NOT ask questions.
Do NOT add text before or after JSON.
Output must start with { and end with }.
If unsure, still output valid JSON.
""")

            return self.ask_structured_question(
                message=prompt,
                model=model,
                response_model=RepairRankingOutput,
            )

        finally:
            self.response_format = old_format   # <- restore

    def clear_context(self) -> None:
        """Clear the chat context."""
        self.chat_context = []
