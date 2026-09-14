"""
LLM call tool

Provides a unified LLM call interface, supporting OpenAI, Qwen, OpenRouter, and Gemini.
Lazy initialization of the client to avoid side effects when importing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx
import openai

from vis_interact.config import settings

logger = logging.getLogger(__name__)

# ── Lazy client (thread-safe) ────────────────────────────────────────────────────

_openai_client: Optional[openai.Client] = None
_qwen_client: Optional[openai.Client] = None
_openrouter_client: Optional[openai.Client] = None
_gemini_client: Optional[openai.Client] = None
_client_lock = threading.Lock()

_LLM_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _get_client(model: str) -> Tuple[openai.Client, str]:
    """Route to the corresponding API client based on the model name (lazy initialization, thread-safe).

    gpt-*   → OpenAI API (proxy)
    qwen-*  → Qwen (DashScope) API
    */*     → OpenRouter (provider/model format with /)
    gemini-* → Gemini OpenAI compatible API
    """
    global _openai_client, _qwen_client, _openrouter_client, _gemini_client

    with _client_lock:
        if "/" in model:
            if _openrouter_client is None:
                _openrouter_client = openai.Client(
                    api_key=settings.apis.openrouter.api_key,
                    base_url=settings.apis.openrouter.base_url,
                    timeout=_LLM_TIMEOUT,
                )
            return _openrouter_client, "openrouter"

        if model.startswith("gpt"):
            if _openai_client is None:
                _openai_client = openai.Client(
                    api_key=settings.apis.openai.api_key,
                    base_url=settings.apis.openai.base_url,
                    timeout=_LLM_TIMEOUT,
                )
            return _openai_client, "openai"

        if model.startswith("qwen"):
            if _qwen_client is None:
                _qwen_client = openai.Client(
                    api_key=settings.apis.qwen.api_key,
                    base_url=settings.apis.qwen.base_url,
                    timeout=_LLM_TIMEOUT,
                )
            return _qwen_client, "qwen"

        if model.startswith("gemini"):
            if _gemini_client is None:
                _gemini_client = openai.Client(
                    api_key=settings.apis.gemini.api_key,
                    base_url=settings.apis.gemini.base_url,
                    timeout=_LLM_TIMEOUT,
                )
            return _gemini_client, "gemini"

        raise ValueError(f"Unknown model prefix, cannot route to client: {model}")


# ── Response data class ────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """LLM call response"""
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[list] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = False
    refused: bool = False


@dataclass
class LLMLogprobsResponse:
    """LLM call response (with token logprobs)"""
    content: Optional[str] = None
    token_logprobs: Optional[list] = None
    tool_calls: Optional[list] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = False


MAX_TOKENS = 8192


def generate_reply_api(
    messages: list,
    model: str = "",
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    response_format: Optional[dict] = None,
    max_retries: int = 3,
    retry_delay: float = 3.0,
    log_prefix: str = "",
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """Generic LLM call function.

    Args:
        timeout: Timeout seconds for a single HTTP request, overriding the client default value.
        max_tokens: Maximum number of tokens to generate. Default 8192 for Gemini models if not specified.
    """
    if not model:
        model = settings.models.default_model
    lp = f"{log_prefix} " if log_prefix else ""

    client, client_type = _get_client(model)

    if max_tokens is None:
        max_tokens = MAX_TOKENS

    n_msgs = len(messages)
    has_images = any(
        isinstance(m.get("content"), list) and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in m["content"]
        )
        for m in messages
    )
    req_tag = f"model={model} msgs={n_msgs} img={'Y' if has_images else 'N'}"
    if timeout is not None:
        req_tag += f" timeout={timeout}s"
    if max_tokens is not None:
        req_tag += f" max_tokens={max_tokens}"

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            logger.info(
                f"{lp}LLM request started [{req_tag} attempt={attempt+1}/{max_retries}]"
            )

            kwargs: dict = dict(messages=messages, model=model)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            if response_format:
                kwargs["response_format"] = response_format
            if timeout is not None:
                kwargs["timeout"] = timeout
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0

            usage = response.usage
            msg = response.choices[0].message
            thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0

            logger.info(
                f"{lp}LLM request completed [{client_type}] "
                f"{elapsed:.1f}s | tokens: {pt}+{ct}={pt+ct} | "
                f"resp_len={len(msg.content or '')}"
            )

            return LLMResponse(
                content=msg.content,
                thinking=thinking,
                tool_calls=msg.tool_calls,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                success=True,
            )

        except openai.BadRequestError as e:
            elapsed = time.time() - t0
            logger.error(
                f"{lp}LLM request rejected (400) {elapsed:.1f}s: {e}"
            )
            return LLMResponse(success=False, refused=True)
        except Exception as e:
            elapsed = time.time() - t0
            err_type = type(e).__name__
            logger.error(
                f"{lp}LLM request failed [{err_type}] "
                f"attempt={attempt+1}/{max_retries} {elapsed:.1f}s: {e}"
            )
            if attempt < max_retries - 1:
                logger.info(f"{lp}Waiting {retry_delay}s before retrying...")
                time.sleep(retry_delay)

    logger.error(f"{lp}All {max_retries} attempts failed [{req_tag}]")
    return LLMResponse(success=False)
