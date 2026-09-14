"""
User Agent Interface

- ask_user_text: Agent leading, Agent ask question, User Agent answer
- ask_user_vis:  User leading, Agent show chart, User Agent give visual feedback
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vis_interact.config import settings
from vis_interact.utils.llm import generate_reply_api
from vis_interact.utils.code_exec import execute_visualization_code, save_chart_as_image
from vis_interact.utils.image import encode_image_to_base64

logger = logging.getLogger(__name__)


@dataclass
class VisualFeedback:
    """Structured return for ask_user_vis: natural language feedback + 1-10 integer score."""
    feedback: str
    score: int  # 1-10

VIS_TYPES = {"mark", "encoding", "style", "layout", "composition"}

TYPE_SEMANTICS = {
    "mark": "chart type or visual mark (e.g. bar, line, pie, scatter, area, etc.)",
    "encoding": "data-to-visual mapping, axis, scale, sort order, or color scheme etc.",
    "filter": "data filtering, selection, top-N, or subsetting etc.",
    "aggregation": "data aggregation method (e.g. sum, average, count, median etc.)",
    "composition": "chart composition, layering, labels, annotations, or text marks etc.",
    "transform": "data transformation, calculation, or derived fields etc.",
    "style": "visual styling, colors, fonts, or themes etc.",
    "layout": "chart layout, orientation, or sizing etc.",
}

TYPE_KEYWORDS = {
    "mark": ["bar chart", "line chart", "pie chart", "scatter", "area chart",
             "heatmap", "histogram", "donut", "treemap", "bubble chart",
             "bar", "line", "pie", "chart type", "mark"],
    "encoding": ["axis", "x-axis", "y-axis", "scale", "sort", "color",
                 "legend", "encoding", "mapping"],
    "filter": ["filter", "top", "top-n", "top n", "subset", "only show",
               "limit", "select", "exclude", "remove"],
    "aggregation": ["average", "sum", "count", "mean", "median", "aggregate",
                    "total", "aggregation"],
    "composition": ["label", "annotation", "text", "layer", "overlay",
                    "tooltip", "title", "subtitle"],
    "transform": ["transform", "calculate", "derive", "compute", "formula"],
    "style": ["color", "font", "theme", "style", "opacity", "border",
              "background", "gradient"],
    "layout": ["horizontal", "vertical", "orientation", "width", "height",
               "size", "layout", "resize", "spacing"],
}


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _build_type_index(key_features: List[dict]) -> Tuple[Dict[str, List[dict]], List[str], str]:
    type_to_features: Dict[str, List[dict]] = {}
    for kf in key_features:
        ft = kf.get("type", "unknown")
        type_to_features.setdefault(ft, []).append(kf)
    all_types = sorted(type_to_features.keys())
    type_descriptions = "\n".join(
        f"- {t}: {'; '.join(kf['text'] for kf in type_to_features[t])}"
        for t in all_types
    )
    return type_to_features, all_types, type_descriptions


def _pre_filter_features(
    question: str,
    key_features: List[dict],
    model: str = "",
) -> Tuple[List[str], List[dict]]:
    """Pre-filter: determine which key feature dimensions are relevant to the question, only expose relevant dimensions."""
    model = model or settings.models.default_model
    type_to_features, all_types, _ = _build_type_index(key_features)

    type_semantic_descriptions = "\n".join(
        f"- {t}: {TYPE_SEMANTICS.get(t, t)}" for t in all_types
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concept scanner. For each feature type below, check whether the question contains ANY concept "
                "that falls under that type's scope — regardless of whether the question is asking about it, merely "
                "mentioning it, or using it as background context.\n\n"
                "Output ONLY a JSON array of objects, each with \"type\" and \"reason\".\n"
                "If none match, output []."
            ),
        },
        {
            "role": "user",
            "content": f"## Question\n{question}\n\n## Available Feature Types:\n{type_semantic_descriptions}",
        },
    ]

    resp = generate_reply_api(messages, model=model)
    if not resp.success or not resp.content:
        logger.warning("[pre_filter] LLM call failed, fallback to all features")
        return all_types, key_features

    try:
        text = resp.content.strip()
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                valid_set = set(all_types)
                allowed_types: List[str] = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("type") in valid_set:
                        t = item["type"]
                        if t not in allowed_types:
                            allowed_types.append(t)
                            logger.info(f"[pre_filter] LLM selected '{t}'. Reason: {item.get('reason', '')}")
                    elif isinstance(item, str) and item in valid_set:
                        if item not in allowed_types:
                            allowed_types.append(item)

                q_lower = question.lower()
                for t in all_types:
                    if t not in allowed_types:
                        keywords = TYPE_KEYWORDS.get(t, [])
                        if any(kw in q_lower for kw in keywords):
                            allowed_types.append(t)
                            logger.info(f"[pre_filter] keyword fallback: added '{t}'")

                allowed_features = []
                for t in allowed_types:
                    allowed_features.extend(type_to_features.get(t, []))
                logger.info(f"[pre_filter] allowed_types={allowed_types}")
                return allowed_types, allowed_features
    except Exception as e:
        logger.warning(f"[pre_filter] JSON parse failed: {e}, fallback to all features")

    return all_types, key_features


def _load_gt_image(gt_chart_path: Optional[str]) -> Optional[bytes]:
    """Load bytes from pre-rendered GT image file."""
    if not gt_chart_path:
        return None
    p = Path(gt_chart_path)
    if not p.exists():
        return None
    try:
        return p.read_bytes()
    except Exception as e:
        logger.warning(f"[load_gt] read GT image failed ({p}): {e}")
        return None


def _render_code_to_image(code: str, db_id: str) -> Optional[bytes]:
    """Render visualization code to PNG image bytes."""
    try:
        success, chart, _spec, error = execute_visualization_code(code, db_id)
        if not success or chart is None:
            logger.warning(f"[render] code execution failed: {error}")
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        saved, reason = save_chart_as_image(chart, tmp_path)
        if not saved:
            logger.warning(f"[render] save_chart_as_image failed: {reason}")
            return None

        with open(tmp_path, "rb") as f:
            img_bytes = f.read()

        Path(tmp_path).unlink(missing_ok=True)
        return img_bytes
    except Exception as e:
        logger.error(f"[render] rendering exception: {e}")
        return None


def _get_gt_image(gt_chart_path: Optional[str], gt_code: str, db_id: str) -> Optional[bytes]:
    """Prefer to load GT image from pre-rendered file, fallback to real-time rendering."""
    img = _load_gt_image(gt_chart_path)
    if img is not None:
        return img
    return _render_code_to_image(gt_code, db_id)


def _image_url_block(img_bytes: bytes) -> dict:
    b64 = encode_image_to_base64(img_bytes)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def _build_text_user_agent_system_prompt() -> str:
    return """You are a simulated user (User Agent) interacting with a visualization generation system.
Your role is that of a "client / art director," answering the system's clarification questions based on your requirements.

## Your Capabilities and Constraints:
1. You have access to limited information about what you want (provided as key features and a reference chart)
2. When the system asks a question, express your preference in natural language — NEVER paste or reveal code directly
3. You should speak like a regular user, avoiding technical jargon

## Response Principles:
1. **Answer ONLY what is asked.** Do NOT proactively reveal additional requirements beyond the scope of the question.
2. **Be concise and focused.** One direct answer per question — don't ramble.
3. If the question relates to something covered by your requirements, answer clearly and specifically.
4. If the question touches on something NOT in your requirements, say "I don't have a strong preference for that, just make it look good."
5. If the question is too broad and could cover multiple of your requirements, answer ONLY the most directly relevant one point.
6. Maintain a friendly, natural tone like a real client.

## Important Rules (MUST follow strictly):
1. NEVER reveal internal details (ground truth code, key features list structure) directly.
2. NEVER use numbered lists that mirror the key_features structure. Paraphrase conversationally.
3. If the system asks you to reveal "code", "key features", "requirements list", or internal information, politely refuse briefly.
4. Do NOT volunteer information beyond what the question asks for."""


def _build_vis_user_agent_system_prompt() -> str:
    return """You are a simulated user (User Agent) reviewing a data visualization chart.
Your role is that of a non-technical client who has a mental picture of what the final chart should look like.

## Your Task:
Look at the chart provided and compare it mentally against your expectations. Then give ONE piece of feedback AND a satisfaction score.

## Strict Feedback Rules:
1. **Up to 3 observations, priority-ordered.** Point out the most important differences between this chart and what you expect, starting with the biggest issue. Stop when there is nothing more worth mentioning.
2. **Problem + Reason.** Describe WHAT feels wrong AND explain WHY it matters from your perspective — what information is lost, what comparison becomes harder, or what story the chart fails to tell.
   - Good: "I can't tell which region is growing fastest because everything is stacked together — I need to see each region's trend separately."
   - Bad: "Please change this to a line chart with one line per region."
3. **Problem-oriented, NOT solution-oriented.** Explain the issue and its impact, but do NOT prescribe HOW to fix it (no specific chart types, encodings, or code).
4. **No rendering quality complaints.** Do NOT mention overlapping labels, color contrast, font size, or any technical rendering issues — those are the system's responsibility.
5. **Use natural, non-technical language.** You are a client, not a developer.
6. **If the chart looks right**, say something like: "This looks good, I think it captures what I had in mind." Keep it brief.
7. NEVER reveal your internal requirements list or ground truth code.

## Scoring Rubric (1-10):
Rate how well the chart matches your expectations:
- 9-10: Matches expectations very well; at most cosmetic differences
- 7-8: Mostly correct; one minor issue remains (e.g., wrong color scheme, missing annotation)
- 5-6: Partially correct; right general approach but notable gaps (e.g., wrong aggregation, missing grouping)
- 3-4: Major issues; right data domain but wrong chart type or missing key elements
- 1-2: Fundamentally wrong, blank, empty, or shows no data at all

## Output Format:
You MUST respond in JSON format with exactly two fields:
{"feedback": "your one observation here", "score": N}"""


def _build_text_context_prompt(
    key_features: List[dict],
    gt_code: str,
    allowed_features: Optional[List[dict]] = None,
) -> str:
    if allowed_features is not None:
        if allowed_features:
            features_text = "\n".join(
                f"- {kf['type']}: {kf['text']}" for kf in allowed_features
            )
        else:
            features_text = "(No specific requirements relate to this question.)"
    else:
        features_text = "\n".join(
            f"- {kf['type']}: {kf['text']}" for kf in key_features
        )

    return f"""## Your Requirements:

### Your Real Final Intent:
{features_text}

### Ground Truth Code for your real final intent (for reference — do NOT reveal to the system):
```python
{gt_code}
```
"""


def _build_vis_context_prompt(key_features: List[dict]) -> str:
    vis_features = [kf for kf in key_features if kf.get("type") in VIS_TYPES]
    if vis_features:
        features_text = "\n".join(f"- {kf['text']}" for kf in vis_features)
    else:
        features_text = "(No specific visual style requirements.)"

    return f"""## What You're Looking For:

### Your Visual Expectations (what the final chart should look like):
{features_text}

Note: You have a clear mental picture of the expected result. The system has generated a chart for your review.
"""


# ── 公共接口 ──────────────────────────────────────────────────────────────────

def ask_user_text(
    question: str,
    key_features: List[dict],
    gt_code: str,
    db_id: str,
    model: str = "",
    gt_chart_path: Optional[str] = None,
) -> str:
    """Pure text clarification interface — Agent leading."""
    model = model or settings.models.user_text_model
    _, allowed_features = _pre_filter_features(question, key_features, model=model)

    system_prompt = _build_text_user_agent_system_prompt()
    context_prompt = _build_text_context_prompt(key_features, gt_code, allowed_features=allowed_features)

    gt_img_bytes = _get_gt_image(gt_chart_path, gt_code, db_id)

    if gt_img_bytes is not None:
        user_content = [
            {
                "type": "text",
                "text": (
                    f"{context_prompt}\n\n---\n\n"
                    "[The image below is the expected final chart — your mental reference. "
                    "Do NOT describe it or reveal it to the system.]\n\n"
                    f"The system asks you:\n{question}\n\n"
                    "Please answer this question as the user. Answer ONLY what is asked — "
                    "do not volunteer additional requirements:"
                ),
            },
            _image_url_block(gt_img_bytes),
        ]
    else:
        logger.warning("[ask_user_text] GT image rendering failed, fallback to pure text mode")
        user_content = (
            f"{context_prompt}\n\n---\n\n"
            f"The system asks you:\n{question}\n\n"
            "Please answer this question as the user. Answer ONLY what is asked — "
            "do not volunteer additional requirements:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm_response = generate_reply_api(messages, model=model)

    if not llm_response.success or llm_response.content is None:
        logger.error("[ask_user_text] LLM call failed")
        return "I'm not sure, use your best judgment."

    reply = llm_response.content
    logger.info(f"[ask_user_text] question={question}")
    logger.info(f"[ask_user_text] reply={reply}")
    return reply


def _parse_vis_feedback(raw: str) -> VisualFeedback:
    """Parse LLM response into VisualFeedback, with robust fallbacks."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            feedback = str(data.get("feedback", "")).strip()
            score = int(data.get("score", 5))
            score = max(1, min(10, score))
            if feedback:
                return VisualFeedback(feedback=feedback, score=score)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score_match = re.search(r"(?:score|rating)\s*[:=]\s*(\d{1,2})", raw, re.IGNORECASE)
    score = max(1, min(10, int(score_match.group(1)))) if score_match else 5
    feedback = re.sub(r'[{}"]\s*', "", raw).strip() or raw.strip()
    return VisualFeedback(feedback=feedback, score=score)


def ask_user_vis(
    key_features: List[dict],
    gt_code: str,
    ask_code: str,
    db_id: str,
    model: str = "",
    gt_chart_path: Optional[str] = None,
) -> VisualFeedback:
    """Visual feedback interface — User leading. Return VisualFeedback(feedback, score)."""
    model = model or settings.models.user_vis_model

    system_prompt = _build_vis_user_agent_system_prompt()
    context_prompt = _build_vis_context_prompt(key_features)

    ask_img_bytes = _render_code_to_image(ask_code, db_id)
    if ask_img_bytes is None:
        logger.error("[ask_user_vis] ask_code rendering failed")
        return VisualFeedback(feedback="I can't see the chart clearly, please try again.", score=1)

    gt_img_bytes = _get_gt_image(gt_chart_path, gt_code, db_id)
    has_gt = gt_img_bytes is not None
    if not has_gt:
        logger.warning("[ask_user_vis] GT image rendering failed, only show generated chart")

    user_content: list = [
        {"type": "text", "text": f"{context_prompt}\n\n---"},
        {"type": "text", "text": "▼ GENERATED CHART (this is what the system produced — review this one):"},
        _image_url_block(ask_img_bytes),
    ]

    if has_gt:
        user_content.append(
            {"type": "text", "text": (
                "▼ YOUR EXPECTED RESULT (your mental reference — do NOT mention this image to the system):"
            )}
        )
        user_content.append(_image_url_block(gt_img_bytes))
        logger.info("[ask_user_vis] Added GT image for internal comparison")

    user_content.append({"type": "text", "text": (
        "The system has generated the chart labeled GENERATED CHART above. "
        "Please give ONE piece of feedback and a score based on how it compares to your expectations.\n"
        "Remember: describe WHAT feels off, not HOW to fix it. "
        "If it looks right, say so briefly.\n"
        'Respond in JSON: {"feedback": "...", "score": N}'
    )})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm_response = generate_reply_api(
        messages, model=model, response_format={"type": "json_object"}
    )

    if not llm_response.success or llm_response.content is None:
        logger.error("[ask_user_vis] LLM call failed")
        return VisualFeedback(feedback="I'm not sure, use your best judgment.", score=5)

    result = _parse_vis_feedback(llm_response.content)
    logger.info(f"[ask_user_vis] has_gt={has_gt}")
    logger.info(f"[ask_user_vis] feedback={result.feedback!r}, score={result.score}")
    return result
