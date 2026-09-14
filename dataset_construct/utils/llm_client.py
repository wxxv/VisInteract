"""
LLM/VLM client for Vis-Interact Dataset Construction.

Supports separate LLM and VLM configurations:
- LLM: Used for text-only tasks (code generation, SQL transformation, etc.)
- VLM: Used for vision tasks (chart validation with image input)
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import inspect
import copy

from openai import OpenAI

from core.config import get_config, LLMConfig, VLMConfig

logger = logging.getLogger(__name__)
# llm_io_logger removed to avoid global state issues in multithreading
# We will write directly to handlers instead


class RateLimitError(Exception):
    """API rate limit error (429)"""
    pass

# Cache for candidate-specific log handlers (to avoid recreating)
_candidate_log_handlers: Dict[str, logging.FileHandler] = {}


def _get_or_create_llm_io_handler() -> Optional[logging.FileHandler]:
    """
    Get or create llm_io handler for current candidate context.
    
    Returns handler specific to current candidate if context is set,
    otherwise returns/creates a run-level handler.
    """
    try:
        from .logging_context import get_llm_io_path
        
        llm_io_path = get_llm_io_path()
        if not llm_io_path:
            return None
        
        # Use path as key for handler cache
        handler_key = str(llm_io_path)
        
        # Return cached handler if exists
        if handler_key in _candidate_log_handlers:
            return _candidate_log_handlers[handler_key]
        
        # Create new handler for this path
        llm_io_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(llm_io_path, encoding="utf-8", mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Cache it
        _candidate_log_handlers[handler_key] = handler
        
        return handler
        
    except Exception as e:
        logger.debug(f"Failed to create candidate-specific llm_io handler: {e}")
        return None


def _log_to_llm_io(message: str):
    """
    Log message to current candidate's llm_io file.
    Thread-safe: Gets specific handler for current context and writes directly.
    """
    handler = _get_or_create_llm_io_handler()
    if handler:
        # Create a LogRecord manually to use the handler's formatting/locking
        record = logging.LogRecord(
            name="llm_io",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        handler.emit(record)


def _infer_process_name() -> str:
    """Infer a process name from call stack when caller doesn't provide one."""
    try:
        stack = inspect.stack()
        for frame_info in stack[2:]:
            module = inspect.getmodule(frame_info.frame)
            mod_name = module.__name__ if module else ""
            if mod_name.endswith("llm_client") or mod_name.endswith("dataset_construct.llm_client"):
                continue
            fn = frame_info.function
            return f"{mod_name}.{fn}" if mod_name else fn
    except Exception:
        pass
    return "unknown_process"


def _truncate_text(text: str, max_chars: int) -> Dict[str, Any]:
    if text is None:
        return {"text": None, "truncated": False, "original_length": 0}
    original_length = len(text)
    if original_length <= max_chars:
        return {"text": text, "truncated": False, "original_length": original_length}
    return {
        "text": text[:max_chars] + "\n...[truncated]...",
        "truncated": True,
        "original_length": original_length,
    }


def _md_escape_fence(text: str) -> str:
    """Avoid accidentally closing markdown fences."""
    if text is None:
        return ""
    # Replace triple backticks with a visually similar variant to avoid accidentally closing markdown fences
    return str(text).replace("```", "``\u200b`")


def _format_messages_markdown(messages: List[Dict[str, Any]]) -> str:
    """Render chat messages as readable markdown."""
    lines: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        lines.append(f"## Message {i+1} ({role})")
        if isinstance(content, list):
            # VLM: mixed image_url + text
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img = item.get("image_url", {})
                    detail = img.get("detail")
                    url_len = img.get("url_length")
                    lines.append(f"- image_url: omitted (detail={detail}, url_length={url_len})")
                elif isinstance(item, dict) and item.get("type") == "text":
                    lines.append("")
                    lines.append(_md_escape_fence(str(item.get("text", ""))))
                else:
                    lines.append(f"- {str(item)}")
        else:
            lines.append("")
            lines.append(_md_escape_fence(str(content or "")))
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_llm_io_markdown(
    *,
    ts: str,
    process: str,
    model: str,
    usage: Dict[str, Any],
    temperature: Any,
    response_format: Any,
    messages: List[Dict[str, Any]],
    response_text: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Format a single LLM I/O record as markdown."""
    meta_str = ""
    if metadata:
        meta_str = json.dumps(metadata, ensure_ascii=False)

    usage_str = json.dumps(usage, ensure_ascii=False)
    rf_str = json.dumps(response_format, ensure_ascii=False, indent=2) if response_format else "null"

    header = [
        f"# process: {process}",
        f"- ts: {ts}",
        f"- model: {model}",
        f"- usage: `{usage_str}`",
        f"- temperature: `{temperature}`",
        f"- response_format: `{rf_str}`",
    ]
    if meta_str:
        header.append(f"- metadata: `{meta_str}`")

    resp = response_text.get("text", "")
    trunc = response_text.get("truncated", False)
    orig_len = response_text.get("original_length", 0)

    md_parts = []
    md_parts.append("\n".join(header))
    md_parts.append("")
    md_parts.append("# Request Messages")
    md_parts.append(_format_messages_markdown(messages))
    md_parts.append("")
    md_parts.append("# Response Content")
    md_parts.append(f"- truncated: `{trunc}` (original_length={orig_len})")
    md_parts.append("")
    md_parts.append(_md_escape_fence(str(resp)))
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")
    return "\n".join(md_parts)


def _sanitize_messages_for_logging(messages: List[Dict[str, Any]], max_chars: int = 20000) -> List[Dict[str, Any]]:
    """
    Sanitize messages for logging:
    - remove/shorten base64 image urls
    - truncate long text blobs
    """
    safe = copy.deepcopy(messages)
    for msg in safe:
        content = msg.get("content")
        # VLM uses content as a list of items with image_url/text
        if isinstance(content, list):
            new_items = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img = item.get("image_url", {})
                    url = img.get("url", "")
                    # Do not log base64 payload
                    img_meta = {
                        "detail": img.get("detail"),
                        "url": "[omitted_image_url]",
                        "url_length": len(url) if isinstance(url, str) else 0,
                    }
                    new_items.append({"type": "image_url", "image_url": img_meta})
                elif isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    new_items.append({"type": "text", "text": _truncate_text(str(t), max_chars)["text"]})
                else:
                    new_items.append(item)
            msg["content"] = new_items
        elif isinstance(content, str):
            msg["content"] = _truncate_text(content, max_chars)["text"]
    return safe


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    usage: Dict[str, int]
    model: str
    
    def parse_json(self) -> Optional[Dict[str, Any]]:
        """Try to parse response as JSON."""
        try:
            # Extract JSON from potentially mixed content
            json_str = _extract_json_from_response(self.content)
            if not json_str:
                return None
            
            return json.loads(json_str)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"JSON parse failed: {e}, content preview: {self.content[:200] if self.content else 'empty'}")
            return None


def _extract_json_from_response(content: str) -> Optional[str]:
    """
    Extract JSON content from LLM response.
    Handles various formats including:
    - Direct JSON
    - JSON in markdown code blocks
    - JSON mixed with thinking/reasoning content
    """
    if not content:
        return None
    
    content = content.strip()
    
    # Remove markdown code block markers
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove language tag and closing fence
        if len(lines) > 2:
            content = "\n".join(lines[1:-1]).strip()
    
    # Try direct parse first
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass
    
    # Find JSON by matching braces/brackets
    # Walk through string to find balanced braces
    def find_balanced_json(s: str, start_char: str) -> Optional[str]:
        """Find balanced JSON object/array starting with start_char."""
        end_char = '}' if start_char == '{' else ']'
        start_idx = s.find(start_char)
        if start_idx == -1:
            return None
        
        count = 0
        in_string = False
        escape = False
        
        for i in range(start_idx, len(s)):
            c = s[i]
            
            # Handle string escaping
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if c == start_char:
                count += 1
            elif c == end_char:
                count -= 1
                if count == 0:
                    return s[start_idx:i+1]
        
        return None
    
    # Try to find JSON object first
    json_str = find_balanced_json(content, '{')
    if json_str:
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array
    json_str = find_balanced_json(content, '[')
    if json_str:
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    return content


class LLMClient:
    """
    Client for LLM/VLM API calls.
    
    Supports separate LLM and VLM configurations:
    - LLM client: Used for text-only tasks
    - VLM client: Used for vision tasks (image input)
    
    The routing is automatic based on input type.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        vlm_config: Optional[VLMConfig] = None
    ):
        config = get_config()
        self.llm_config = llm_config or config.llm
        self.vlm_config = vlm_config or config.vlm
        
        # Create LLM client for text-only tasks
        self.llm_client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url
        )
        
        # Create VLM client for vision tasks
        # May use same endpoint or different one based on config
        if (self.vlm_config.api_key == self.llm_config.api_key and 
            self.vlm_config.base_url == self.llm_config.base_url):
            # Same endpoint, reuse client
            self.vlm_client = self.llm_client
        else:
            # Different endpoint, create separate client
            self.vlm_client = OpenAI(
                api_key=self.vlm_config.api_key,
                base_url=self.vlm_config.base_url
            )
        
        # Keep backward compatibility with self.config
        self.config = self.llm_config
    
    # ------------------------------------------------------------------
    # Helpers for model routing
    # ------------------------------------------------------------------
    @staticmethod
    def _is_chat_model(model: str) -> bool:
        """Heuristic: codex models use completions, others use chat."""
        return "codex" not in model.lower()
    
    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        """Flatten chat messages to a single prompt for completions."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # concatenate text parts
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text", "")))
                content = "\n".join(texts)
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, str]] = None,
        process_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> LLMResponse:
        """
        Send chat completion request using LLM (text-only) with exponential backoff retry.
        
        For vision tasks with image input, use complete_with_image() instead.
        
        Note: Some models (e.g., gpt-5.1-codex) don't support temperature or
        json_object response_format. These are handled automatically.
        
        Raises:
            RateLimitError: If 429 error persists after all retries
        """
        # _ensure_llm_io_logger() - Removed, using direct logging
        proc = process_name or _infer_process_name()
        target_model = self.llm_config.model
        temp = self.llm_config.temperature
        
        def _log_and_build_response(resp_model: str, content: str, usage_obj: Any):
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            try:
                safe_messages = _sanitize_messages_for_logging(messages)
                md = _format_llm_io_markdown(
                    ts=datetime.now().isoformat(),
                    process=proc,
                    model=resp_model,
                    usage=usage,
                    temperature=temp,
                    response_format=response_format,
                    messages=safe_messages,
                    response_text=_truncate_text(content, 20000),
                    metadata=metadata or {},
                )
                _log_to_llm_io(md)
            except Exception:
                pass
            return LLMResponse(content=content, usage=usage, model=resp_model)
        
        # Build kwargs - some models don't support temperature
        kwargs = {
            "model": target_model,
            "messages": messages,
        }
        
        # Only add temperature if model supports it (codex models don't)
        if self._is_chat_model(target_model):
            kwargs["temperature"] = temp
        
        # Only add response_format if provided
        if response_format:
            kwargs["response_format"] = response_format
        
        # Add thinking/reasoning parameters (model-specific)
        thinking_kwargs = self.llm_config.get_thinking_kwargs()
        kwargs.update(thinking_kwargs)
        
        logger.debug(f"LLM chat request: model={target_model}, temp={temp}, thinking={self.llm_config.enable_thinking}, params={thinking_kwargs}")
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return _log_and_build_response(response.model, content, response.usage)
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for 429 rate limit error
                if "429" in str(e) or "rate_limit" in error_str or "rate limit" in error_str:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    # If this is the last attempt, raise RateLimitError
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"Rate limit exceeded after {max_retries} retries") from e
                    
                    # Exponential backoff: 1s, 2s, 4s...
                    wait_time = (2 ** attempt) * 1.0
                    logger.info(f"Backing off for {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Other errors - log and retry with shorter wait
                    logger.error(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        # Last attempt failed with non-rate-limit error
                        raise
        
        # Should never reach here, but just in case
        raise Exception("LLM request failed after all retries")
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Simple completion with optional system prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
    
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Complete and parse response as JSON."""
        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            **kwargs
        )
        return response.parse_json()
    
    def analyze_code(self, code: str, task: str) -> LLMResponse:
        """Analyze code for a specific task."""
        system_prompt = """You are an expert code analyst specializing in data visualization.
Analyze the given code and respond to the task precisely."""
        
        prompt = f"""## Code to Analyze
```python
{code}
```

## Task
{task}
"""
        return self.complete(prompt, system_prompt=system_prompt, process_name="llm_client.analyze_code")
    
    def generate_altair_code(
        self,
        data_description: str,
        visualization_requirement: str,
        key_features: List[Dict[str, Any]],
        chart_type: str
    ) -> str:
        """Generate Altair visualization code."""
        system_prompt = """You are an expert in Altair/Vega-Lite visualizations.
Generate clean, working Altair code based on the requirements.
Only output the Python code, no explanations."""
        
        key_features_str = json.dumps(key_features, indent=2)
        
        prompt = f"""## Data Description
{data_description}

## Visualization Requirement
{visualization_requirement}

## Target Chart Type
{chart_type}

## Key Features (must satisfy)
{key_features_str}

Generate complete Altair code that satisfies all key features.
The code should:
1. Import necessary libraries (altair, pandas)
2. Assume data is in a DataFrame called 'df'
3. Create the chart and assign it to 'chart'
4. End with 'chart' (not chart.show())
"""
        response = self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=self.llm_config.temperature,
            process_name="llm_client.generate_altair_code"
        )
        
        # Extract code from response
        content = response.content.strip()
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0]
        elif "```" in content:
            code = content.split("```")[1].split("```")[0]
        else:
            code = content
        
        return code.strip()
    
    def rewrite_instruction(
        self,
        original_instruction: str,
        target_visualization: str,
        ambiguity_points: List[str]
    ) -> str:
        """Rewrite SQL instruction to visualization question with ambiguity."""
        system_prompt = """You are an expert in creating natural language questions for data visualization.
Rewrite the given instruction into a visualization-focused question that:
1. Retains the core business intent
2. Shifts focus from "getting results" to "visualizing trends/comparisons/distributions"
3. Deliberately omits or makes ambiguous the specified points
4. Sounds natural and like a real user request"""
        
        prompt = f"""## Original Instruction
{original_instruction}

## Target Visualization Type
{target_visualization}

## Points to Make Ambiguous (do NOT specify these clearly)
{json.dumps(ambiguity_points, indent=2)}

Rewrite as a natural visualization request question.
Output only the rewritten question, nothing else.
"""
        response = self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=self.llm_config.temperature,
            process_name="llm_client.rewrite_instruction"
        )
        return response.content.strip()
    
    def extract_key_features(self, vega_lite_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key features from Vega-Lite spec."""
        system_prompt = """You are an expert in Altair/Vega-Lite specifications.
Extract the most important features from the spec as path-op-value constraints.
Focus on: mark type, encodings (field, type, aggregate), transforms, compositions, interactions."""
        
        prompt = f"""## Vega-Lite Spec
{json.dumps(vega_lite_spec, indent=2)}

Extract key features as a JSON array of objects with:
- path: attribute path (e.g., "mark.type", "encoding.x.field")
- op: operator (exists, eq, in, contains)
- value: expected value

Focus on features that define the visualization's meaning, not styling details.
Output only the JSON array.
"""
        result = self.complete_json(
            prompt,
            system_prompt=system_prompt,
            temperature=self.llm_config.temperature,
            process_name="llm_client.extract_key_features"
        )
        return result.get("features", []) if result else []
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """
        Extract JSON from a response that may contain explanation text.
        
        Handles formats like:
        - Pure JSON: {...}
        - Markdown code block: ```json {...} ```
        - Text + code block: "Some explanation... ```json {...} ```"
        """
        import re
        
        if not content:
            return None
        
        content = content.strip()
        
        # Try to find JSON in markdown code blocks (```json or ```)
        # Pattern matches ```json or ``` followed by content and closing ```
        code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        matches = re.findall(code_block_pattern, content)
        
        for match in matches:
            match = match.strip()
            # Check if this looks like JSON (starts with { or [)
            if match.startswith('{') or match.startswith('['):
                try:
                    json.loads(match)  # Validate it's parseable
                    return match
                except json.JSONDecodeError:
                    continue
        
        # Try to find raw JSON object in the text (not in code blocks)
        # Look for {...} pattern
        json_obj_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_obj_pattern, content)
        
        # Try each match, starting from the longest (most complete)
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # If content itself starts with { or [, it might be pure JSON
        if content.startswith('{') or content.startswith('['):
            return content
        
        return None
    
    def complete_with_image(
        self,
        prompt: str,
        image_base64: str,
        image_media_type: str = "image/png",
        system_prompt: Optional[str] = None,
        process_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Complete with image input using VLM (Vision Language Model) with exponential backoff retry.
        
        This method automatically uses the VLM client and model for vision tasks.
        
        Args:
            prompt: Text prompt for analysis
            image_base64: Base64 encoded image data
            image_media_type: MIME type (image/png, image/jpeg, etc.)
            system_prompt: Optional system prompt
            max_retries: Maximum number of retry attempts for rate limits
        
        Returns:
            Parsed JSON response or None if failed
            
        Raises:
            RateLimitError: If 429 error persists after all retries
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build content with text and image
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_media_type};base64,{image_base64}",
                    "detail": self.vlm_config.image_detail  # Image detail level from config
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        messages.append({"role": "user", "content": content})
        
        # Use VLM client and model for vision tasks
        vlm_model = self.vlm_config.model
        vlm_temp = self.vlm_config.temperature
        
        # _ensure_llm_io_logger() - Removed
        proc = process_name or _infer_process_name()
        
        logger.debug(f"VLM request: model={vlm_model}, temp={vlm_temp}")
        
        # Build kwargs - some models don't support temperature
        vlm_kwargs = {
            "model": vlm_model,
            "messages": messages,
        }
        
        # Only add temperature if model supports it
        if self._is_chat_model(vlm_model):
            vlm_kwargs["temperature"] = vlm_temp
        
        # Add thinking/reasoning parameters (model-specific)
        thinking_kwargs = self.vlm_config.get_thinking_kwargs()
        vlm_kwargs.update(thinking_kwargs)
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.vlm_client.chat.completions.create(**vlm_kwargs)
                
                result_content = response.choices[0].message.content.strip()

                # Log I/O (Markdown) - do not log base64 image
                try:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                    safe_messages = _sanitize_messages_for_logging(messages)
                    md = _format_llm_io_markdown(
                        ts=datetime.now().isoformat(),
                        process=proc,
                        model=response.model,
                        usage=usage,
                        temperature=vlm_temp,
                        response_format=None,
                        messages=safe_messages,
                        response_text=_truncate_text(result_content, 20000),
                        metadata={
                            **(metadata or {}),
                            "image_media_type": image_media_type,
                            "image_base64_length": len(image_base64) if isinstance(image_base64, str) else 0,
                            "image_detail": self.vlm_config.image_detail,
                            "model_type": "vlm",
                        },
                    )
                    _log_to_llm_io(md)
                except Exception as log_err:
                    logger.debug(f"VLM logging failed: {log_err}")
                
                # Parse JSON from response - handle various formats
                json_content = self._extract_json_from_response(result_content)
                if json_content:
                    return json.loads(json_content)
                
                # Try parsing the raw content as JSON
                return json.loads(result_content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error in VLM response: {e}")
                return None
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for 429 rate limit error
                if "429" in str(e) or "rate_limit" in error_str or "rate limit" in error_str:
                    logger.warning(f"VLM rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    # If this is the last attempt, raise RateLimitError
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"VLM rate limit exceeded after {max_retries} retries") from e
                    
                    # Exponential backoff: 1s, 2s, 4s...
                    wait_time = (2 ** attempt) * 1.0
                    logger.info(f"Backing off for {wait_time}s before VLM retry...")
                    time.sleep(wait_time)
                else:
                    # Other errors - log and return None
                    logger.error(f"VLM completion error: {e}")
                    return None
        
        # Should never reach here
        return None
    
    def complete_with_images(
        self,
        prompt: str,
        images: List[str],
        image_media_type: str = "image/png",
        system_prompt: Optional[str] = None,
        process_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        response_format: Optional[str] = "json_object",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Complete with multiple image inputs using VLM (Vision Language Model).
        
        This method automatically uses the VLM client and model for vision tasks.
        Supports multiple images in a single request.
        
        Args:
            prompt: Text prompt for analysis
            images: List of base64-encoded image strings
            image_media_type: MIME type for all images (image/png, image/jpeg, etc.)
            system_prompt: Optional system prompt
            process_name: Process name for logging
            metadata: Additional metadata for logging
            max_retries: Maximum number of retry attempts for rate limits
            response_format: Response format ("json_object" or None)
        
        Returns:
            Parsed JSON response or None if failed
            
        Raises:
            RateLimitError: If 429 error persists after all retries
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build content with text and multiple images
        content = [{"type": "text", "text": prompt}]
        
        for image_base64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_media_type};base64,{image_base64}",
                    "detail": self.vlm_config.image_detail  # Image detail level from config
                }
            })
        
        messages.append({"role": "user", "content": content})
        
        # Use VLM client and model for vision tasks
        vlm_model = self.vlm_config.model
        vlm_temp = self.vlm_config.temperature
        
        proc = process_name or _infer_process_name()
        
        logger.debug(f"VLM request with {len(images)} images: model={vlm_model}, temp={vlm_temp}")
        
        # Build kwargs
        vlm_kwargs = {
            "model": vlm_model,
            "messages": messages,
        }
        
        # Add temperature if model supports it
        if self._is_chat_model(vlm_model):
            vlm_kwargs["temperature"] = vlm_temp
        
        # Add response format if specified
        if response_format:
            vlm_kwargs["response_format"] = {"type": response_format}
        
        # Add thinking/reasoning parameters (model-specific)
        thinking_kwargs = self.vlm_config.get_thinking_kwargs()
        vlm_kwargs.update(thinking_kwargs)
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.vlm_client.chat.completions.create(**vlm_kwargs)
                
                result_content = response.choices[0].message.content.strip()

                # Log I/O (Markdown) - do not log base64 images
                try:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                    safe_messages = _sanitize_messages_for_logging(messages)
                    
                    # Calculate total image data size
                    total_image_size = sum(len(img) for img in images if isinstance(img, str))
                    
                    md = _format_llm_io_markdown(
                        ts=datetime.now().isoformat(),
                        process=proc,
                        model=response.model,
                        usage=usage,
                        temperature=vlm_temp,
                        response_format={"type": response_format} if response_format else None,
                        messages=safe_messages,
                        response_text=_truncate_text(result_content, 20000),
                        metadata={
                            **(metadata or {}),
                            "image_media_type": image_media_type,
                            "num_images": len(images),
                            "total_image_base64_length": total_image_size,
                            "image_detail": self.vlm_config.image_detail,
                            "model_type": "vlm",
                        },
                    )
                    _log_to_llm_io(md)
                except Exception as log_err:
                    logger.debug(f"VLM logging failed: {log_err}")
                
                # Parse JSON from response - handle various formats
                json_content = self._extract_json_from_response(result_content)
                if json_content:
                    return json.loads(json_content)
                
                # Try parsing the raw content as JSON
                return json.loads(result_content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error in VLM response: {e}")
                return None
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for 429 rate limit error
                if "429" in str(e) or "rate_limit" in error_str or "rate limit" in error_str:
                    logger.warning(f"VLM rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    # If this is the last attempt, raise RateLimitError
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"VLM rate limit exceeded after {max_retries} retries") from e
                    
                    # Exponential backoff: 1s, 2s, 4s...
                    wait_time = (2 ** attempt) * 1.0
                    logger.info(f"Backing off for {wait_time}s before VLM retry...")
                    time.sleep(wait_time)
                else:
                    # Other errors - log and return None
                    logger.error(f"VLM completion error: {e}")
                    return None
        
        # Should never reach here
        return None


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
