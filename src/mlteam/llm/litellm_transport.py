"""LiteLLM transport: provider-agnostic structured-output calls.

LiteLLM (https://docs.litellm.ai) proxies 100+ providers behind one
`completion()` call, so the same agents run on Anthropic, OpenAI, Bedrock,
Azure, local models, etc. — the model string carries the provider
(e.g. "anthropic/claude-sonnet-4-6", "openai/gpt-4o", "ollama/llama3").

Structured output is requested via `response_format` JSON-schema mode when the
provider supports it (LiteLLM advertises this per-model); otherwise the schema
is embedded in the system prompt as a universal fallback. Either way, the
LLMClient still validates the returned JSON against the Pydantic schema and
retries on failure, so a provider that ignores the hint degrades to
prompt-driven JSON rather than breaking.

Install with the optional extra:  uv sync --extra litellm
"""

from __future__ import annotations

import json
import time

import structlog

from mlteam.llm.pricing import TokenUsage
from mlteam.llm.transport import LLMRawResponse, LLMRefusalError, LLMRequest

_log = structlog.get_logger(__name__)

_SCHEMA_INSTRUCTION = (
    "\n\nYou MUST respond with a single JSON object that validates against this "
    "JSON Schema. Output only the JSON — no prose, no code fences:\n{schema}"
)


class LiteLLMTransport:
    """Structured-output transport over litellm.completion.

    ``model`` is a LiteLLM model string including the provider prefix. Extra
    keyword args (api_base, api_key, custom_llm_provider, ...) pass straight
    through to litellm.completion, so any provider LiteLLM supports works
    without code changes.
    """

    def __init__(
        self,
        *,
        use_response_format: bool = True,
        drop_params: bool = True,
        **completion_kwargs: object,
    ) -> None:
        import litellm  # imported lazily; optional dependency

        self._litellm = litellm
        self._use_response_format = use_response_format
        self._completion_kwargs = dict(completion_kwargs)
        # Let LiteLLM silently drop params a given provider doesn't accept,
        # so one call site works across heterogeneous providers.
        litellm.drop_params = drop_params

    def _supports_response_format(self, model: str) -> bool:
        if not self._use_response_format:
            return False
        try:
            return bool(
                self._litellm.supports_response_schema(model=model)
                or self._litellm.supports_response_format(model=model)
            )
        except Exception:  # pragma: no cover - depends on litellm model map
            return False

    def complete(self, request: LLMRequest) -> LLMRawResponse:
        native = self._supports_response_format(request.model)
        user_content = request.user_content
        kwargs = dict(self._completion_kwargs)
        if native:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "schema": request.output_schema,
                    "strict": True,
                },
            }
        else:
            user_content += _SCHEMA_INSTRUCTION.format(schema=json.dumps(request.output_schema))

        started = time.perf_counter()
        response = self._litellm.completion(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": user_content},
            ],
            **kwargs,
        )
        latency = time.perf_counter() - started

        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise LLMRefusalError("provider content filter declined the request")
        text = choice.message.content or ""
        usage = _extract_usage(response)
        model_name = getattr(response, "model", None) or request.model
        return LLMRawResponse(text=text, usage=usage, model=model_name, latency_seconds=latency)


def _extract_usage(response: object) -> TokenUsage:
    """Normalize LiteLLM's usage (OpenAI-shaped) into our TokenUsage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    # Some providers surface cache reads via prompt_tokens_details.
    cache_read = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cache_read = int(getattr(details, "cached_tokens", 0) or 0)
    return TokenUsage(
        input_tokens=max(prompt - cache_read, 0),
        output_tokens=completion,
        cache_read_input_tokens=cache_read,
    )
