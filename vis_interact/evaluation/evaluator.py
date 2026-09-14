"""
Evaluator module — Based on double Judge system to evaluate the generated visualization code

1. Code LLM Judge：Judge KeyFeature from code level
2. Chart VLM Judge：Judge KeyFeature from image level
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from vis_interact.config import settings
from vis_interact.utils.code_exec import execute_visualization_code, save_chart_as_image
from vis_interact.utils.image import encode_image_to_base64 as _encode_image_to_base64
from vis_interact.utils.llm import generate_reply_api

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def loads_judge_json(text: str):
    """Parse a judge response as JSON, tolerating markdown code fences.

    Some providers ignore `response_format={"type": "json_object"}` and still
    wrap the payload in a ```json fence.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))

    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class KeyFeatureJudgment:
    """Single KeyFeature single-view judgment result"""
    feature_id: str
    satisfied: bool
    confidence: float
    reasoning: str
    eval_success: bool = False


@dataclass
class KeyFeatureResult:
    """Complete evaluation result for a single KeyFeature (double-view)"""
    feature_id: str
    feature_text: str
    feature_type: str
    is_must: bool

    code_judgment: Optional[KeyFeatureJudgment] = None
    chart_judgment: Optional[KeyFeatureJudgment] = None

    @property
    def satisfied_code(self) -> bool:
        return self.code_judgment.satisfied if self.code_judgment else False

    @property
    def satisfied_chart(self) -> bool:
        return self.chart_judgment.satisfied if self.chart_judgment else False

    @property
    def satisfied_merge(self) -> bool:
        return self.satisfied_code and self.satisfied_chart


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    sample_id: str

    key_feature_results: List[KeyFeatureResult] = field(default_factory=list)

    kf_total: int = 0
    kf_must_total: int = 0

    code_kf_pass_count: int = 0
    code_must_pass_count: int = 0
    code_strict_success: bool = False

    chart_kf_pass_count: int = 0
    chart_must_pass_count: int = 0
    chart_strict_success: bool = False

    merge_kf_pass_count: int = 0
    merge_must_pass_count: int = 0
    merge_strict_success: bool = False

    code_eval_failed_kf_ids: List[str] = field(default_factory=list)
    chart_eval_failed_kf_ids: List[str] = field(default_factory=list)

    @property
    def code_kf_pass_rate(self) -> float:
        return self.code_kf_pass_count / self.kf_total if self.kf_total > 0 else 0.0

    @property
    def code_must_pass_rate(self) -> float:
        return self.code_must_pass_count / self.kf_must_total if self.kf_must_total > 0 else 0.0

    @property
    def chart_kf_pass_rate(self) -> float:
        return self.chart_kf_pass_count / self.kf_total if self.kf_total > 0 else 0.0

    @property
    def chart_must_pass_rate(self) -> float:
        return self.chart_must_pass_count / self.kf_must_total if self.kf_must_total > 0 else 0.0

    @property
    def merge_kf_pass_rate(self) -> float:
        return self.merge_kf_pass_count / self.kf_total if self.kf_total > 0 else 0.0

    @property
    def merge_must_pass_rate(self) -> float:
        return self.merge_must_pass_count / self.kf_must_total if self.kf_must_total > 0 else 0.0

    def compute_scores(self):
        if not self.key_feature_results:
            return

        self.kf_total = len(self.key_feature_results)
        self.kf_must_total = sum(1 for kf in self.key_feature_results if kf.is_must)

        code_must_pass = True
        chart_must_pass = True
        merge_must_pass = True

        for kf in self.key_feature_results:
            code_j = kf.code_judgment
            chart_j = kf.chart_judgment

            code_eval_success = code_j.eval_success if code_j else False
            if not code_eval_success:
                self.code_eval_failed_kf_ids.append(kf.feature_id)
            if code_eval_success and kf.satisfied_code:
                self.code_kf_pass_count += 1
                if kf.is_must:
                    self.code_must_pass_count += 1
            elif kf.is_must:
                code_must_pass = False

            chart_eval_success = chart_j.eval_success if chart_j else False
            if not chart_eval_success:
                self.chart_eval_failed_kf_ids.append(kf.feature_id)
            if chart_eval_success and kf.satisfied_chart:
                self.chart_kf_pass_count += 1
                if kf.is_must:
                    self.chart_must_pass_count += 1
            elif kf.is_must:
                chart_must_pass = False

            merge_eval_success = code_eval_success and chart_eval_success
            if merge_eval_success and kf.satisfied_merge:
                self.merge_kf_pass_count += 1
                if kf.is_must:
                    self.merge_must_pass_count += 1
            elif kf.is_must:
                merge_must_pass = False

        self.code_strict_success = code_must_pass
        self.chart_strict_success = chart_must_pass
        self.merge_strict_success = merge_must_pass


CHART_IMAGE_MAX_SIZE = 1280


# ── Tool functions ──────────────────────────────────────────────────────────────────

def encode_image_to_base64(
    image_path: str,
    max_size: Optional[int] = None,
) -> Optional[str]:
    try:
        return _encode_image_to_base64(image_path, max_size=max_size)
    except Exception as e:
        logger.warning(f"Image encoding failed: {e}")
        return None


# ── Code LLM Judge ──────────────────────────────────────────────────────────────────

def code_llm_judge(
    code: str,
    key_features: List[dict],
    vis_question: str,
    gt_code: Optional[str] = None,
    model: str = "",
    max_retries: int = 3,
    log_prefix: str = "",
) -> List[KeyFeatureJudgment]:
    model = model or settings.models.eval_model
    tag = f"{log_prefix} [Code Judge]" if log_prefix else "[Code Judge]"

    system_prompt = """You are an expert Code LLM Judge for data visualization evaluation.
Your task is to check if a generated Altair/Vega-Lite visualization code satisfies specific key features.

## Important
The user's original request is provided only as background context. It may be vague or ambiguous and does NOT fully represent the user's true intent. The key features are the ground-truth specification of what the user actually wants. A ground truth reference code is also provided to help you understand the intended implementation. Always judge strictly against the key features, not your own interpretation of the original request.

## Evaluation Principles:
1. Only judge based on PROVABLE facts from the code - do not guess or assume
2. If uncertain, return satisfied=false with lower confidence
3. Consider the code logic; use the ground truth code as a reference for what correct behavior looks like
4. Be strict but fair - multiple implementation approaches may satisfy the same feature

## Output Format:
For each key feature, output a JSON object with exactly 4 keys: id, reasoning, satisfied, confidence.
Write "reasoning" BEFORE "satisfied" — analyze first, then conclude."""

    features_text = "\n".join([
        f"{i+1}. id: {kf.get('id', f'kf_{i}')}, type: {kf.get('type', 'unknown')}, text: {kf['text']}"
        for i, kf in enumerate(key_features)
    ])

    gt_section = ""
    if gt_code:
        gt_section = f"\n## Ground Truth Code (Reference)\n```python\n{gt_code}\n```\n"

    user_prompt = f"""## User Request (Background)
{vis_question}

## Key Features to Check
{features_text}
{gt_section}
## Generated Code (To Evaluate)
```python
{code}
```

## Instructions
For each key feature, compare the generated code against the KEY FEATURE TEXT (not the user's original request).
```json
{{
  "results": [
    {{
      "id": "<feature_id>",
      "reasoning": "<cite relevant code and explain whether it satisfies this key feature>",
      "satisfied": true/false,
      "confidence": 0.0-1.0
    }}
  ]
}}
```

Output ONLY the JSON object, no other text."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(f"{tag} Start evaluating {len(key_features)} features, model={model}")
    judge_t0 = time.time()

    results: List[KeyFeatureJudgment] = []
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        llm_response = generate_reply_api(
            messages, model, temperature=0.0,
            response_format={"type": "json_object"},
            log_prefix=tag,
        )
        if not llm_response.success:
            logger.warning(
                f"{tag} LLM call failed (attempt {attempt}/{max_retries}), "
                f"time: {time.time() - judge_t0:.1f}s"
            )
            for i, kf in enumerate(key_features):
                results.append(KeyFeatureJudgment(
                    feature_id=kf.get("id", f"kf_{i}"),
                    satisfied=False, confidence=0.0,
                    reasoning="Code Judge LLM call failed", eval_success=False,
                ))
            return results

        response_text = llm_response.content or ""
        try:
            parsed = loads_judge_json(response_text)
            judge_results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(judge_results, list):
                returned_ids = [jr.get("id") for jr in judge_results if isinstance(jr, dict)]
                expected_ids = [kf.get("id", f"kf_{i}") for i, kf in enumerate(key_features)]
                logger.info(f"{tag} returned feature ids: {returned_ids}")
                logger.info(f"{tag} expected feature ids: {expected_ids}")
                unmatched = set(expected_ids) - set(returned_ids)
                if unmatched:
                    logger.warning(f"{tag} unmatched feature ids: {unmatched}")
                    if attempt < max_retries:
                        missing_str = ", ".join(sorted(unmatched))
                        logger.info(f"{tag} some features are missing, trigger retry (attempt {attempt}/{max_retries})")
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content":
                            f'Your response has malformed JSON or is missing results for these features: {missing_str}. '
                            f'Output a COMPLETE JSON object with ALL {len(expected_ids)} features: '
                            '{"results": [{"id": "...", "reasoning": "...", "satisfied": true/false, "confidence": 0.0-1.0}]}'
                        })
                        last_error = f"Partial results: missing {unmatched}"
                        results.clear()
                        continue
                for i, kf in enumerate(key_features):
                    expected_id = kf.get("id", f"kf_{i}")
                    judgment_data = next(
                        (jr for jr in judge_results if isinstance(jr, dict) and jr.get("id") == expected_id),
                        None,
                    )
                    if judgment_data:
                        missing = [k for k in ("id", "reasoning", "satisfied", "confidence") if k not in judgment_data]
                        results.append(KeyFeatureJudgment(
                            feature_id=expected_id,
                            satisfied=judgment_data.get("satisfied", False),
                            confidence=judgment_data.get("confidence", 0.0),
                            reasoning=judgment_data.get("reasoning", f"Missing keys: {missing}"),
                            eval_success=not missing,
                        ))
                    else:
                        logger.warning(f"{tag} feature '{expected_id}' not found in returned results (final result)")
                        results.append(KeyFeatureJudgment(
                            feature_id=expected_id, satisfied=False, confidence=0.0,
                            reasoning="Judge did not return result for this feature", eval_success=False,
                        ))
                n_pass = sum(1 for r in results if r.satisfied)
                n_ok = sum(1 for r in results if r.eval_success)
                logger.info(
                    f"{tag} completed: {n_pass}/{len(results)} satisfied, "
                    f"{n_ok}/{len(results)} eval_success, "
                    f"time: {time.time() - judge_t0:.1f}s (attempt={attempt})"
                )
                return results
            last_error = "Parsed JSON is not a list or object with 'results' key"
            logger.warning(f"{tag} JSON structure exception (attempt {attempt}/{max_retries}), type: {type(judge_results)}")
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            logger.warning(f"{tag} JSON parse failed (attempt {attempt}/{max_retries}): {e}")
            logger.debug(f"{tag} original response: {response_text}")

        if attempt < max_retries:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": 'Format error. Output ONLY a JSON object: {"results": [{"id": "...", "reasoning": "...", "satisfied": true/false, "confidence": 0.0-1.0}]}'})

    logger.error(
        f"{tag} evaluation failed, retried {max_retries} times, "
        f"time: {time.time() - judge_t0:.1f}s: {last_error}"
    )
    return [
        KeyFeatureJudgment(
            feature_id=kf.get("id", f"kf_{i}"), satisfied=False, confidence=0.0,
            reasoning="Code Judge evaluation failed", eval_success=False,
        )
        for i, kf in enumerate(key_features)
    ]


# ── Chart VLM Judge ───────────────────────────────────────────────────────────

def chart_vlm_judge(
    chart_image_path: str,
    key_features: List[dict],
    vis_question: str,
    gt_chart_image_path: Optional[str] = None,
    model: str = "",
    max_retries: int = 3,
    log_prefix: str = "",
) -> List[KeyFeatureJudgment]:
    model = model or settings.models.eval_model
    tag = f"{log_prefix} [Chart Judge]" if log_prefix else "[Chart Judge]"

    pred_image_b64 = encode_image_to_base64(chart_image_path, max_size=CHART_IMAGE_MAX_SIZE)
    if not pred_image_b64:
        logger.error(f"{tag} cannot read image: {chart_image_path}")
        return [
            KeyFeatureJudgment(
                feature_id=kf.get("id", f"kf_{i}"), satisfied=False, confidence=0.0,
                reasoning="Image exists but failed to load.", eval_success=True,
            )
            for i, kf in enumerate(key_features)
        ]

    system_prompt = """You are an expert Chart VLM Judge for data visualization evaluation.
Your task is to check if a rendered visualization chart satisfies specific key features based on visual inspection.

## Important
The key features are the ground-truth specification. Always judge strictly against the key features.

## Evaluation Principles:
1. Judge based on what you can SEE in the chart image
2. CRITICAL: If the chart appears empty or only shows axes without data marks, ALL features must be satisfied=false.
3. If uncertain, return satisfied=false with lower confidence

## Output Format:
For each key feature, output a JSON object with exactly 4 keys: id, reasoning, satisfied, confidence.
Write "reasoning" BEFORE "satisfied"."""

    features_text = "\n".join([
        f"{i+1}. id: {kf.get('id', f'kf_{i}')}, type: {kf.get('type', 'unknown')}, text: {kf['text']}"
        for i, kf in enumerate(key_features)
    ])

    user_content = [
        {
            "type": "text",
            "text": f"""## User Request (Background)
{vis_question}

## Key Features to Check
{features_text}

## Instructions
```json
{{
  "results": [
    {{
      "id": "<feature_id>",
      "reasoning": "<describe what you observe>",
      "satisfied": true/false,
      "confidence": 0.0-1.0
    }}
  ]
}}
```

Output ONLY the JSON object, no other text.

The first image is the chart to evaluate:""",
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pred_image_b64}"}},
    ]

    # if gt_chart_image_path:
    #     gt_image_b64 = encode_image_to_base64(gt_chart_image_path, max_size=CHART_IMAGE_MAX_SIZE)
    #     if gt_image_b64:
    #         user_content[0]["text"] += "\n\nThe second image is the ground truth reference chart:"
    #         user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gt_image_b64}"}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    has_gt = gt_chart_image_path is not None
    logger.info(
        f"{tag} start evaluating {len(key_features)} features, "
        f"model={model}, has_gt_image={has_gt}"
    )
    judge_t0 = time.time()

    results: List[KeyFeatureJudgment] = []
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        llm_response = generate_reply_api(
            messages, model,
            temperature=0.0,
            response_format={"type": "json_object"},
            log_prefix=tag,
        )
        if not llm_response.success:
            elapsed = time.time() - judge_t0
            if llm_response.refused:
                logger.warning(
                    f"{tag} API rejected request (image content problem), time: {elapsed:.1f}s, "
                    f"all features failed"
                )
                return [
                    KeyFeatureJudgment(
                        feature_id=kf.get("id", f"kf_{i}"),
                        satisfied=False, confidence=1.0,
                        reasoning="Chart image rejected by API (content/format issue)",
                        eval_success=True,
                    )
                    for i, kf in enumerate(key_features)
                ]
            logger.warning(
                f"{tag} LLM call failed (attempt {attempt}/{max_retries}), "
                f"time: {elapsed:.1f}s"
            )
            for i, kf in enumerate(key_features):
                results.append(KeyFeatureJudgment(
                    feature_id=kf.get("id", f"kf_{i}"),
                    satisfied=False, confidence=0.0,
                    reasoning="Chart Judge LLM call failed", eval_success=False,
                ))
            return results

        response_text = llm_response.content or ""
        try:
            parsed = loads_judge_json(response_text)
            judge_results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(judge_results, list):
                returned_ids = [jr.get("id") for jr in judge_results if isinstance(jr, dict)]
                expected_ids = [kf.get("id", f"kf_{i}") for i, kf in enumerate(key_features)]
                logger.info(f"{tag} returned feature ids: {returned_ids}")
                logger.info(f"{tag} expected feature ids: {expected_ids}")
                unmatched = set(expected_ids) - set(returned_ids)
                if unmatched:
                    logger.warning(f"{tag} unmatched feature ids: {unmatched}")
                    extra = set(returned_ids) - set(expected_ids)
                    if extra:
                        logger.warning(f"{tag} returned extra ids: {extra}")
                    if attempt < max_retries:
                        missing_str = ", ".join(sorted(unmatched))
                        logger.info(f"{tag} some features are missing, trigger retry (attempt {attempt}/{max_retries})")
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": [{"type": "text", "text":
                            f'Your response has malformed JSON or is missing results for these features: {missing_str}. '
                            f'Output a COMPLETE JSON object with ALL {len(expected_ids)} features: '
                            '{"results": [{"id": "...", "reasoning": "...", "satisfied": true/false, "confidence": 0.0-1.0}]}'
                        }]})
                        last_error = f"Partial results: missing {unmatched}"
                        results.clear()
                        continue
                for i, kf in enumerate(key_features):
                    expected_id = kf.get("id", f"kf_{i}")
                    judgment_data = next(
                        (jr for jr in judge_results if isinstance(jr, dict) and jr.get("id") == expected_id),
                        None,
                    )
                    if judgment_data:
                        missing = [k for k in ("id", "reasoning", "satisfied", "confidence") if k not in judgment_data]
                        results.append(KeyFeatureJudgment(
                            feature_id=expected_id,
                            satisfied=judgment_data.get("satisfied", False),
                            confidence=judgment_data.get("confidence", 0.0),
                            reasoning=judgment_data.get("reasoning", f"Missing keys: {missing}"),
                            eval_success=not missing,
                        ))
                    else:
                        logger.warning(f"{tag} feature '{expected_id}' not found in returned results (final result)")
                        results.append(KeyFeatureJudgment(
                            feature_id=expected_id, satisfied=False, confidence=0.0,
                            reasoning="Judge did not return result for this feature", eval_success=False,
                        ))
                n_pass = sum(1 for r in results if r.satisfied)
                n_ok = sum(1 for r in results if r.eval_success)
                logger.info(
                    f"{tag} completed: {n_pass}/{len(results)} satisfied, "
                    f"{n_ok}/{len(results)} eval_success, "
                    f"time: {time.time() - judge_t0:.1f}s (attempt={attempt})"
                )
                return results
            last_error = "Parsed JSON is not a list or object with 'results' key"
            logger.warning(f"{tag} JSON structure exception (attempt {attempt}/{max_retries}), type: {type(judge_results)}")
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            logger.warning(f"{tag} JSON parse failed (attempt {attempt}/{max_retries}): {e}")
            logger.debug(f"{tag} original response: {response_text}")

        if attempt < max_retries:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": [{"type": "text", "text": 'Format error. Output ONLY a JSON object: {"results": [{"id": "...", "reasoning": "...", "satisfied": true/false, "confidence": 0.0-1.0}]}'}]})

    logger.error(
        f"{tag} evaluation failed, retried {max_retries} times, "
        f"time: {time.time() - judge_t0:.1f}s: {last_error}"
    )
    return [
        KeyFeatureJudgment(
            feature_id=kf.get("id", f"kf_{i}"), satisfied=False, confidence=0.0,
            reasoning="Chart Judge evaluation failed", eval_success=False,
        )
        for i, kf in enumerate(key_features)
    ]


# ── Evaluator class ──────────────────────────────────────────────────────────────────

class Evaluator:
    """Double Judge evaluator"""

    def __init__(
        self,
        code_judge_model: str = "",
        chart_judge_model: str = "",
    ):
        self.code_judge_model = code_judge_model or settings.models.eval_model
        self.chart_judge_model = chart_judge_model or settings.models.eval_model

    def evaluate(
        self,
        sample: dict,
        pred_image_path: Optional[str] = None,
        gt_image_path: Optional[str] = None,
        skip_chart_judge: bool = False,
    ) -> EvaluationResult:
        sid = sample["sample_id"]
        result = EvaluationResult(sample_id=sid)
        eval_t0 = time.time()

        generated_code = sample.get("final_code")
        if not generated_code:
            logger.warning(f"[{sid}] sample has no final_code, skip evaluation")
            return result

        key_features = sample.get("key_features", [])
        if not key_features:
            logger.info(f"[{sid}] has no KeyFeatures, skip evaluation")
            return result

        vis_question = sample.get("initial_question", "")
        gt_code = sample.get("ground_truth_code")

        logger.info(
            f"[{sid}] start evaluating: {len(key_features)} features, "
            f"has_pred_img={pred_image_path is not None}, "
            f"has_gt_img={gt_image_path is not None}, "
            f"skip_chart={skip_chart_judge}"
        )

        sample_tag = f"[{sid}]"

        code_t0 = time.time()
        logger.info(f"[{sid}] >>> Code LLM Judge start")
        code_judgments = code_llm_judge(
            code=generated_code, key_features=key_features, vis_question=vis_question,
            gt_code=gt_code, model=self.code_judge_model, log_prefix=sample_tag,
        )
        code_elapsed = time.time() - code_t0
        logger.info(f"[{sid}] <<< Code LLM Judge end, time: {code_elapsed:.1f}s")

        chart_judgments = []
        if pred_image_path and not skip_chart_judge:
            chart_t0 = time.time()
            logger.info(f"[{sid}] >>> Chart VLM Judge start")
            chart_judgments = chart_vlm_judge(
                chart_image_path=pred_image_path, key_features=key_features, vis_question=vis_question,
                gt_chart_image_path=gt_image_path, model=self.chart_judge_model, log_prefix=sample_tag,
            )
            chart_elapsed = time.time() - chart_t0
            logger.info(f"[{sid}] <<< Chart VLM Judge end, time: {chart_elapsed:.1f}s")
        else:
            reason = "Chart Judge skipped" if skip_chart_judge else "No chart image available"
            logger.info(f"[{sid}] Chart Judge skipped: {reason}")
            for i, kf in enumerate(key_features):
                chart_judgments.append(KeyFeatureJudgment(
                    feature_id=kf.get("id", f"kf_{i}"), satisfied=False, confidence=1.0,
                    reasoning=reason, eval_success=False,
                ))

        for i, kf in enumerate(key_features):
            kf_result = KeyFeatureResult(
                feature_id=kf.get("id", f"kf_{i}"),
                feature_text=kf["text"],
                feature_type=kf.get("type", "unknown"),
                is_must=kf.get("must", True),
                code_judgment=code_judgments[i] if i < len(code_judgments) else None,
                chart_judgment=chart_judgments[i] if i < len(chart_judgments) else None,
            )
            result.key_feature_results.append(kf_result)

        result.compute_scores()

        total_elapsed = time.time() - eval_t0
        logger.info(
            f"[{sid}] evaluation completed (total time: {total_elapsed:.1f}s): "
            f"Code={result.code_kf_pass_count}/{result.kf_total} (strict={result.code_strict_success}), "
            f"Chart={result.chart_kf_pass_count}/{result.kf_total} (strict={result.chart_strict_success}), "
            f"Merge={result.merge_kf_pass_count}/{result.kf_total} (strict={result.merge_strict_success})"
        )

        return result
