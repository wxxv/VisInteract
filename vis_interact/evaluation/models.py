"""
评估结果数据模型

定义 runner.py 批量评估所需的全部数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KeyFeatureDetail:
    """单个 KeyFeature 的评估详情"""
    feature_id: str
    feature_text: str
    feature_type: str
    is_must: bool

    code_satisfied: bool
    code_confidence: float
    code_reasoning: str
    code_eval_success: bool

    chart_satisfied: bool
    chart_confidence: float
    chart_reasoning: str
    chart_eval_success: bool

    merge_satisfied: bool


@dataclass
class EvalSampleResult:
    """单个样本的评估结果"""
    sample_id: str

    kf_total: int = 0
    kf_must_total: int = 0

    code_kf_pass_count: int = 0
    code_must_pass_count: int = 0
    code_kf_pass_rate: float = 0.0
    code_must_pass_rate: float = 0.0
    code_strict_success: bool = False

    chart_kf_pass_count: int = 0
    chart_must_pass_count: int = 0
    chart_kf_pass_rate: float = 0.0
    chart_must_pass_rate: float = 0.0
    chart_strict_success: bool = False

    merge_kf_pass_count: int = 0
    merge_must_pass_count: int = 0
    merge_kf_pass_rate: float = 0.0
    merge_must_pass_rate: float = 0.0
    merge_strict_success: bool = False

    all_eval_success: bool = False
    error_message: Optional[str] = None
    eval_time: float = 0.0

    code_eval_failed_kf_ids: List[str] = field(default_factory=list)
    chart_eval_failed_kf_ids: List[str] = field(default_factory=list)

    key_feature_details: List[KeyFeatureDetail] = field(default_factory=list)


@dataclass
class EvalResults:
    """整批次评估的汇总结果"""
    track: str = ""
    model: str = ""
    eval_model: str = ""
    timestamp: str = ""
    result_dir: str = ""

    total_samples: int = 0
    evaluated_samples: int = 0
    valid_samples: int = 0
    valid_kf_total: int = 0
    valid_kf_must_total: int = 0

    code_renderable_count: int = 0
    code_renderable_rate: float = 0.0

    code_dataset_score: float = 0.0
    code_must_dataset_score: float = 0.0
    code_strict_success_rate: float = 0.0

    chart_dataset_score: float = 0.0
    chart_must_dataset_score: float = 0.0
    chart_strict_success_rate: float = 0.0

    merge_dataset_score: float = 0.0
    merge_must_dataset_score: float = 0.0
    merge_strict_success_rate: float = 0.0

    eval_failed_sample_ids: List[str] = field(default_factory=list)
    sample_results: List[EvalSampleResult] = field(default_factory=list)
