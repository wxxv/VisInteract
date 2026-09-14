"""
Validator for Step 8: Quality validation.
Validates samples for data quality, visualization correctness, and ambiguity effectiveness.
Includes VLM-based visual validation for rendered chart images.
"""
import base64
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from core.models import (
    VisSample, Feature, AmbiguityProfile, ValidationResult, IterationFeedback
)
from utils.chart_contracts import get_contract_manager
from generation.spec import get_spec_generator
from utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Optional imports for chart rendering
try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

try:
    import vl_convert as vlc
    HAS_VLC = True
except ImportError:
    HAS_VLC = False


# Failure codes as defined in the design doc
class FailureCode:
    EXEC_FAIL = "EXEC_FAIL"           # SQL/pandas execution failure
    SEMANTIC_DRIFT = "SEMANTIC_DRIFT"  # Deviated from semantic anchor
    KF_MISMATCH = "KF_MISMATCH"        # Key features not satisfied
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"  # Chart contract not met
    INTERACT_DEGENERATE = "INTERACT_DEGENERATE"  # Interactive element doesn't work
    VIS_UNREADABLE = "VIS_UNREADABLE"  # Visual readability issues
    AMB_WEAK = "AMB_WEAK"              # Ambiguity injection too weak
    AMB_STRONG = "AMB_STRONG"          # Ambiguity injection too strong


@dataclass
class DataValidationReport:
    """Report from data validation phase."""
    passed: bool = True
    row_count: int = 0
    column_count: int = 0
    null_percentage: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "null_percentage": self.null_percentage,
            "issues": self.issues
        }


@dataclass
class VisValidationReport:
    """Report from visualization validation phase."""
    passed: bool = True
    key_features_satisfied: int = 0
    key_features_total: int = 0
    contract_satisfied: bool = True
    interaction_valid: bool = True
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "key_features_satisfied": self.key_features_satisfied,
            "key_features_total": self.key_features_total,
            "contract_satisfied": self.contract_satisfied,
            "interaction_valid": self.interaction_valid,
            "issues": self.issues
        }


@dataclass
class VLMValidationReport:
    """Report from VLM-based visual validation."""
    passed: bool = True
    image_generated: bool = False
    vlm_verified: bool = False
    readability_score: float = 0.0
    data_representation_correct: bool = True
    visual_issues: List[str] = field(default_factory=list)
    vlm_feedback: str = ""
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "image_generated": self.image_generated,
            "vlm_verified": self.vlm_verified,
            "readability_score": self.readability_score,
            "data_representation_correct": self.data_representation_correct,
            "visual_issues": self.visual_issues,
            "vlm_feedback": self.vlm_feedback,
            "image_path": self.image_path
        }


class Validator:
    """Validates samples for quality."""
    
    # Thresholds
    MIN_ROWS = 5
    MAX_ROWS = 1000
    MAX_NULL_PERCENTAGE = 0.5
    MIN_CATEGORIES_FOR_PIE = 2
    MAX_CATEGORIES_FOR_PIE = 8
    MIN_READABILITY_SCORE = 0.7
    
    def __init__(self, enable_vlm_validation: bool = True):
        self.contract_manager = get_contract_manager()
        self.spec_generator = get_spec_generator()
        self.llm = get_llm_client()
        self.enable_vlm_validation = enable_vlm_validation and HAS_ALTAIR and HAS_VLC
    
    def validate_sample(self, sample: VisSample) -> ValidationResult:
        """Perform full validation on a sample."""
        logger.debug(f"Validating sample: {sample.sample_id}")
        failure_codes = []
        
        # Phase A: Data Validation
        data_report = self.validate_data(
            sample.data,
            sample.chart_type,
            []  # semantic_anchor not available in VisSample
        )
        
        if not data_report.passed:
            failure_codes.extend([
                f"{FailureCode.EXEC_FAIL}: {issue}"
                for issue in data_report.issues
                if "execution" in issue.lower() or "empty" in issue.lower()
            ])
            failure_codes.extend([
                f"{FailureCode.SEMANTIC_DRIFT}: {issue}"
                for issue in data_report.issues
                if "semantic" in issue.lower()
            ])
        
        # Phase B: Visualization Validation (only if data passes)
        vis_report = VisValidationReport()
        if data_report.passed:
            vis_report = self.validate_visualization(
                sample.vega_lite_spec,
                sample.features,
                sample.chart_type
            )
            
            if not vis_report.passed:
                if not vis_report.contract_satisfied:
                    failure_codes.append(FailureCode.CONTRACT_VIOLATION)
                if vis_report.key_features_satisfied < vis_report.key_features_total:
                    failure_codes.append(
                        f"{FailureCode.KF_MISMATCH}: "
                        f"{vis_report.key_features_satisfied}/{vis_report.key_features_total} satisfied"
                    )
                if not vis_report.interaction_valid:
                    failure_codes.append(FailureCode.INTERACT_DEGENERATE)
        
        # Phase C: VLM Visual Validation (only if spec validation passes)
        vlm_report = VLMValidationReport()
        if data_report.passed and vis_report.passed and self.enable_vlm_validation:
            vlm_report = self.validate_with_vlm(
                sample.vega_lite_spec,
                sample.data,
                sample.vis_question_clear or sample.vis_question,
                sample.chart_type,
                sample.features
            )
            
            if not vlm_report.passed:
                for issue in vlm_report.visual_issues:
                    failure_codes.append(f"{FailureCode.VIS_UNREADABLE}: {issue}")
        
        # Phase D: Ambiguity Validation
        amb_issues = self.validate_ambiguity(
            sample.vis_question_clear or sample.vis_question,
            sample.ambiguity_profile,
            []  # semantic_anchor not available in VisSample
        )
        
        for issue in amb_issues:
            if "weak" in issue.lower():
                failure_codes.append(f"{FailureCode.AMB_WEAK}: {issue}")
            elif "strong" in issue.lower():
                failure_codes.append(f"{FailureCode.AMB_STRONG}: {issue}")
        
        passed = len(failure_codes) == 0
        
        # Generate structured iteration feedback
        iteration_feedback = self.generate_iteration_feedback(
            sample, data_report, vis_report, vlm_report, amb_issues
        )
        
        return ValidationResult(
            passed=passed,
            failure_codes=failure_codes,
            data_validation_report=data_report.to_dict(),
            vis_validation_report=vis_report.to_dict(),
            vlm_validation_report=vlm_report.to_dict(),
            suggestions=self._generate_suggestions(failure_codes),
            iteration_feedback=iteration_feedback
        )
    
    def validate_data(
        self,
        data_preview: List[Dict[str, Any]],
        chart_type: str,
        semantic_anchor: List[str]
    ) -> DataValidationReport:
        """Validate data quality and suitability."""
        report = DataValidationReport()
        
        # Convert to DataFrame for analysis
        if not data_preview:
            report.passed = False
            report.issues.append("Data preview is empty")
            return report
        
        df = pd.DataFrame(data_preview)
        report.row_count = len(df)
        report.column_count = len(df.columns)
        
        # Check row count
        if report.row_count < self.MIN_ROWS:
            report.passed = False
            report.issues.append(f"Too few rows: {report.row_count} < {self.MIN_ROWS}")
        
        if report.row_count > self.MAX_ROWS:
            report.issues.append(f"Warning: Many rows: {report.row_count}")
        
        # Check null percentage
        if df.size > 0:
            report.null_percentage = df.isnull().sum().sum() / df.size
            if report.null_percentage > self.MAX_NULL_PERCENTAGE:
                report.passed = False
                report.issues.append(
                    f"High null percentage: {report.null_percentage:.1%}"
                )
        
        # Check chart-specific requirements
        chart_issues = self._validate_chart_requirements(df, chart_type)
        report.issues.extend(chart_issues)
        if chart_issues:
            report.passed = False
        
        # Check semantic anchor coverage
        anchor_issues = self._validate_semantic_anchor(df, semantic_anchor)
        report.issues.extend(anchor_issues)
        
        return report
    
    def _validate_chart_requirements(
        self,
        df: pd.DataFrame,
        chart_type: str
    ) -> List[str]:
        """Validate data meets chart-specific requirements."""
        issues = []
        chart_lower = chart_type.lower()
        
        # Pie/Donut requirements
        if "pie" in chart_lower or "donut" in chart_lower:
            # Check for categorical column
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                unique_values = df[cat_cols[0]].nunique()
                if unique_values < self.MIN_CATEGORIES_FOR_PIE:
                    issues.append(f"Pie chart: too few categories ({unique_values})")
                if unique_values > self.MAX_CATEGORIES_FOR_PIE:
                    issues.append(f"Pie chart: too many categories ({unique_values})")
            
            # Check for negative values
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                if (df[col] < 0).any():
                    issues.append(f"Pie chart: negative values in {col}")
        
        # Histogram requirements
        if "histogram" in chart_lower:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) == 0:
                issues.append("Histogram: no numeric columns")
        
        # Scatter requirements
        if "scatter" in chart_lower:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) < 2:
                issues.append(f"Scatter: need 2+ numeric columns, have {len(num_cols)}")
        
        return issues
    
    def _validate_semantic_anchor(
        self,
        df: pd.DataFrame,
        semantic_anchor: List[str]
    ) -> List[str]:
        """Check if data reflects semantic anchor."""
        issues = []
        
        for anchor in semantic_anchor:
            if anchor.startswith("table:"):
                # Can't verify table from data preview alone
                pass
            elif anchor.startswith("measure:"):
                # Check if measure columns exist
                measure = anchor.split(":", 1)[1]
                # Extract function and field
                if "(" in measure:
                    # e.g., "SUM(revenue)"
                    func_match = measure.split("(")[0].lower()
                    # Just note it - actual verification needs SQL
        
        return issues
    
    def validate_visualization(
        self,
        gt_spec: Dict[str, Any],
        key_features: List[Feature],
        chart_type: str
    ) -> VisValidationReport:
        """Validate visualization spec."""
        report = VisValidationReport()
        report.key_features_total = len(key_features)
        
        # Check key features
        satisfied_count = 0
        for feature in key_features:
            if self.spec_generator._check_feature_in_spec(gt_spec, feature):
                satisfied_count += 1
            else:
                report.issues.append(
                    f"Key feature not satisfied: {feature.path} {feature.op.value}"
                )
        
        report.key_features_satisfied = satisfied_count
        
        # Check contract
        contract = self.contract_manager.build_contract(chart_type)
        contract_violations = self.contract_manager.validate_spec_against_contract(
            gt_spec, contract
        )
        
        if contract_violations:
            report.contract_satisfied = False
            report.issues.extend(contract_violations)
        
        # Check interaction validity for interactive charts
        if self._is_interactive_chart(chart_type):
            report.interaction_valid = self._validate_interaction(gt_spec)
            if not report.interaction_valid:
                report.issues.append("Interactive element may be degenerate")
        
        # Determine overall pass
        report.passed = (
            report.key_features_satisfied >= report.key_features_total * 0.8 and
            report.contract_satisfied and
            report.interaction_valid
        )
        
        return report
    
    def _is_interactive_chart(self, chart_type: str) -> bool:
        """Check if chart type is interactive."""
        interactive_keywords = [
            "interactive", "slider", "brush", "linked",
            "crossfilter", "legend_selection"
        ]
        return any(kw in chart_type.lower() for kw in interactive_keywords)
    
    def _validate_interaction(self, spec: Dict[str, Any]) -> bool:
        """Validate that interaction is not degenerate."""
        params = spec.get("params", [])
        
        if not params:
            return False
        
        # Check that selection is used somewhere
        has_selection_use = False
        
        # Check transforms for filter using selection
        transforms = spec.get("transform", [])
        for t in transforms:
            if "filter" in t:
                filter_val = t["filter"]
                if isinstance(filter_val, dict) and "param" in filter_val:
                    has_selection_use = True
        
        # Check encoding for condition using selection
        encoding = spec.get("encoding", {})
        for channel, enc in encoding.items():
            if isinstance(enc, dict):
                if "condition" in enc:
                    has_selection_use = True
        
        return has_selection_use or len(params) > 0
    
    def validate_with_vlm(
        self,
        gt_spec: Dict[str, Any],
        data_preview: List[Dict[str, Any]],
        initial_question: str,
        chart_type: str,
        key_features: List[Feature]
    ) -> VLMValidationReport:
        """Validate visualization using VLM (Vision Language Model)."""
        report = VLMValidationReport()
        
        # Step 1: Render chart to image
        image_bytes = self._render_spec_to_image(gt_spec, data_preview)
        if image_bytes is None:
            report.passed = False
            report.visual_issues.append("Failed to render chart to image")
            return report
        
        report.image_generated = True
        
        # Step 2: Send to VLM for validation
        vlm_result = self._validate_image_with_vlm(
            image_bytes, initial_question, chart_type, key_features, data_preview
        )
        
        if vlm_result is None:
            report.passed = False
            report.visual_issues.append("VLM validation failed")
            return report
        
        report.vlm_verified = True
        report.readability_score = vlm_result.get("readability_score", 0.0)
        report.data_representation_correct = vlm_result.get("data_correct", True)
        report.vlm_feedback = vlm_result.get("feedback", "")
        
        # Check thresholds
        if report.readability_score < self.MIN_READABILITY_SCORE:
            report.passed = False
            report.visual_issues.append(
                f"Low readability score: {report.readability_score:.2f} < {self.MIN_READABILITY_SCORE}"
            )
        
        if not report.data_representation_correct:
            report.passed = False
            report.visual_issues.append("Data representation incorrect per VLM")
        
        if vlm_result.get("issues"):
            report.visual_issues.extend(vlm_result["issues"])
            if vlm_result["issues"]:
                report.passed = False
        
        return report
    
    def _render_spec_to_image(
        self,
        gt_spec: Dict[str, Any],
        data_preview: List[Dict[str, Any]]
    ) -> Optional[bytes]:
        """Render Vega-Lite spec to PNG image bytes."""
        if not HAS_ALTAIR or not HAS_VLC:
            return None
        
        try:
            # Create spec with data
            spec_with_data = gt_spec.copy()
            spec_with_data["data"] = {"values": data_preview}
            
            # Convert to PNG using vl-convert
            png_bytes = vlc.vegalite_to_png(
                vl_spec=json.dumps(spec_with_data),
                scale=2.0  # Higher resolution for better VLM analysis
            )
            
            return png_bytes
        except Exception as e:
            logger.warning(f"Chart rendering error: {e}")
            return None
    
    def _validate_image_with_vlm(
        self,
        image_bytes: bytes,
        question: str,
        chart_type: str,
        key_features: List[Feature],
        data_preview: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use VLM to validate the rendered chart image."""
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build validation prompt
            key_features_text = "\n".join([
                f"- {f.path}: {f.op.value} {f.value if f.value else ''}"
                for f in key_features[:10]  # Limit to avoid token overflow
            ])
            
            data_sample = json.dumps(data_preview[:3], indent=2, ensure_ascii=False)
            
            prompt = f"""Analyze this data visualization chart and validate its quality.

## User Question
{question}

## Expected Chart Type
{chart_type}

## Key Features Expected
{key_features_text}

## Sample Data
{data_sample}

Please evaluate and respond in JSON format:
{{
    "readability_score": <float 0-1, 1=excellent readability>,
    "data_correct": <bool, whether chart correctly represents the data>,
    "chart_type_match": <bool, whether it matches the expected type>,
    "issues": [<list of specific visual issues found>],
    "feedback": "<brief overall assessment>"
}}

Evaluate:
1. Readability: Are labels, axes, legends clear and not overlapping?
2. Data representation: Does the chart correctly show the data values?
3. Chart type: Does it match the expected visualization type?
4. Visual quality: Any rendering issues, clipping, or artifacts?
"""
            
            # Call VLM with image
            result = self.llm.complete_with_image(
                prompt=prompt,
                image_base64=image_base64,
                image_media_type="image/png",
                process_name="validator.vlm_validate_visualization"
            )
            
            return result
            
        except Exception as e:
            print(f"VLM validation error: {e}")
            return None
    
    def save_validation_image(
        self,
        gt_spec: Dict[str, Any],
        data_preview: List[Dict[str, Any]],
        output_path: Path
    ) -> bool:
        """Save rendered chart image to file for inspection."""
        image_bytes = self._render_spec_to_image(gt_spec, data_preview)
        if image_bytes is None:
            return False
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return True
        except Exception as e:
            print(f"Failed to save image: {e}")
            return False
    
    def validate_ambiguity(
        self,
        initial_question: str,
        ambiguity_profile: AmbiguityProfile,
        semantic_anchor: List[str]
    ) -> List[str]:
        """Validate ambiguity injection effectiveness."""
        issues = []
        question_lower = initial_question.lower()
        
        # Get all ambiguity types (new format) or fallback to single type
        ambiguity_types = getattr(ambiguity_profile, 'ambiguity_types', [])
        if not ambiguity_types:
            # Fallback to old single type
            single_type = getattr(ambiguity_profile, 'ambiguity_type', None)
            if single_type:
                ambiguity_types = [single_type]
        
        # Convert to set of type values for easy checking
        type_values = set(t.value if hasattr(t, 'value') else str(t) for t in ambiguity_types)
        
        # Check for weak ambiguity (too explicit) - for DATA ambiguity
        if "data_ambiguity" in type_values:
            explicit_terms = [
                ("sum", "aggregate"),
                ("average", "aggregate"),
                ("total", "aggregate"),
                ("count", "aggregate"),
                ("by month", "timeUnit"),
                ("by day", "timeUnit"),
                ("by year", "timeUnit"),
                ("top 10", "topk"),
                ("top 5", "topk"),
            ]
            
            for term, feature_type in explicit_terms:
                if term in question_lower:
                    for path in ambiguity_profile.target_feature_paths:
                        if feature_type in path.lower():
                            issues.append(
                                f"Ambiguity too weak: '{term}' explicitly mentioned"
                            )
        
        # Check for VISUALIZATION ambiguity
        if "visualization_ambiguity" in type_values:
            chart_terms = [
                "bar chart", "line chart", "pie chart",
                "scatter plot", "histogram", "heatmap"
            ]
            
            for term in chart_terms:
                if term in question_lower:
                    issues.append(f"Ambiguity too weak: chart type '{term}' explicit")
        
        # Check for strong ambiguity (too vague)
        # Use word count for English, character count for CJK languages
        word_count = len(initial_question.split())
        char_count = len(initial_question)
        is_too_short = word_count < 5 and char_count < 15  # Handle both EN and CJK
        if is_too_short:
            issues.append("Ambiguity too strong: question too short")
        
        if len(ambiguity_profile.target_feature_paths) > 3:
            issues.append("Ambiguity too strong: too many ambiguous points")
        
        # Check semantic anchor preserved
        for anchor in semantic_anchor:
            if anchor.startswith("entity:"):
                entity = anchor.split(":", 1)[1].lower()
                if len(entity) > 3:  # Skip very short entities
                    entity_words = entity.split()
                    if not any(word in question_lower for word in entity_words):
                        issues.append(f"Semantic drift: entity '{entity}' missing")
        
        return issues
    
    def generate_iteration_feedback(
        self,
        sample: VisSample,
        data_report: DataValidationReport,
        vis_report: VisValidationReport,
        vlm_report: VLMValidationReport,
        amb_issues: List[str]
    ) -> IterationFeedback:
        """
        Analyze validation results and generate structured iteration feedback.
        This feedback guides the LLM in fixing issues with SQL or Altair code.
        """
        feedback = IterationFeedback()
        
        sql_needed = False
        code_needed = False
        
        # Analyze data issues → SQL needs iteration
        if not data_report.passed:
            sql_needed = True
            feedback.data_issues = data_report.issues.copy()
            feedback.sql_issues = [
                issue for issue in data_report.issues
                if "row" in issue.lower() or "column" in issue.lower() or 
                   "empty" in issue.lower() or "null" in issue.lower()
            ]
            feedback.sql_suggestions = self._generate_sql_suggestions(data_report)
        
        # Analyze Key Feature issues → Altair code needs iteration
        if vis_report.key_features_satisfied < vis_report.key_features_total:
            code_needed = True
            feedback.missing_features = self._find_missing_features(
                sample.vega_lite_spec, sample.features
            )
            feedback.code_issues.append(
                f"Key features: {vis_report.key_features_satisfied}/{vis_report.key_features_total} satisfied"
            )
            feedback.code_suggestions = self._generate_code_suggestions(vis_report)
        
        # Analyze contract violations → Altair code needs iteration
        if not vis_report.contract_satisfied:
            code_needed = True
            feedback.code_issues.extend([
                issue for issue in vis_report.issues
                if "contract" in issue.lower() or "must have" in issue.lower()
            ])
        
        # Analyze VLM visual issues → Altair code needs iteration
        if vlm_report and not vlm_report.passed:
            code_needed = True
            feedback.visual_issues = vlm_report.visual_issues.copy()
            feedback.visual_suggestions = self._parse_vlm_suggestions(vlm_report)
        
        # Analyze ambiguity issues (usually not code fixable)
        if amb_issues:
            feedback.ambiguity_issues = amb_issues
        
        # Determine iteration target
        if sql_needed and code_needed:
            feedback.iteration_target = "both"
            feedback.priority = "high"
        elif sql_needed:
            feedback.iteration_target = "sql"
            feedback.priority = "high"
        elif code_needed:
            feedback.iteration_target = "altair_code"
            feedback.priority = "medium"
        else:
            feedback.iteration_target = "none"
        
        feedback.needs_iteration = sql_needed or code_needed
        
        return feedback
    
    def _generate_sql_suggestions(self, data_report: DataValidationReport) -> List[str]:
        """Generate SQL fix suggestions based on data issues."""
        suggestions = []
        
        for issue in data_report.issues:
            issue_lower = issue.lower()
            
            if "too few rows" in issue_lower:
                suggestions.append("Remove or relax WHERE conditions to get more data")
                suggestions.append("Consider using UNION to combine related data")
            
            if "empty" in issue_lower:
                suggestions.append("Check table and column names for typos")
                suggestions.append("Verify the WHERE conditions match existing data")
            
            if "null" in issue_lower:
                suggestions.append("Use COALESCE or IFNULL to handle NULL values")
                suggestions.append("Add WHERE column IS NOT NULL to filter nulls")
            
            if "numeric" in issue_lower or "type" in issue_lower:
                suggestions.append("Cast columns to appropriate types using CAST or TRY_CAST")
        
        return list(set(suggestions))
    
    def _generate_code_suggestions(self, vis_report: VisValidationReport) -> List[str]:
        """Generate Altair code fix suggestions based on visualization issues."""
        suggestions = []
        
        for issue in vis_report.issues:
            issue_lower = issue.lower()
            
            if "encoding" in issue_lower or "field" in issue_lower:
                suggestions.append("Check encoding field names match DataFrame columns")
                suggestions.append("Verify encoding types (Q/N/O/T) are correct")
            
            if "color" in issue_lower:
                suggestions.append("Add .encode(color=...) if color encoding is required")
            
            if "aggregate" in issue_lower:
                suggestions.append("Check aggregate functions in encoding (sum, mean, count)")
            
            if "mark" in issue_lower:
                suggestions.append("Verify mark type matches the chart requirements")
            
            if "interaction" in issue_lower or "selection" in issue_lower:
                suggestions.append("Ensure selection is bound and used in conditions or filters")
        
        return list(set(suggestions))
    
    def _find_missing_features(
        self,
        gt_spec: Dict[str, Any],
        key_features: List[Feature]
    ) -> List[Feature]:
        """Find key features not satisfied by the spec."""
        missing = []
        
        for feature in key_features:
            if not self.spec_generator._check_feature_in_spec(gt_spec, feature):
                missing.append(feature)
        
        return missing
    
    def _parse_vlm_suggestions(self, vlm_report: VLMValidationReport) -> List[str]:
        """Extract actionable suggestions from VLM feedback."""
        suggestions = []
        
        # Parse VLM feedback for actionable items
        feedback_lower = vlm_report.vlm_feedback.lower()
        
        if "overlap" in feedback_lower or "overlapping" in feedback_lower:
            suggestions.append("Adjust labelAngle or use labelLimit to prevent overlap")
            suggestions.append("Consider using a horizontal layout or rotating labels")
        
        if "legend" in feedback_lower:
            suggestions.append("Adjust legend position using configure_legend()")
            suggestions.append("Consider using legend=None if legend is not needed")
        
        if "readability" in feedback_lower or "small" in feedback_lower:
            suggestions.append("Increase font size using configure_axis() or configure_title()")
            suggestions.append("Adjust chart dimensions for better readability")
        
        if "color" in feedback_lower and ("contrast" in feedback_lower or "distinguish" in feedback_lower):
            suggestions.append("Use a different color scheme with better contrast")
            suggestions.append("Consider using alt.Scale(scheme='category10') or similar")
        
        for issue in vlm_report.visual_issues:
            if "clipping" in issue.lower() or "cut off" in issue.lower():
                suggestions.append("Add padding or adjust chart size to prevent clipping")
        
        return list(set(suggestions))
    
    def _generate_suggestions(self, failure_codes: List[str]) -> List[str]:
        """Generate suggestions for fixing failures."""
        suggestions = []
        
        for code in failure_codes:
            if FailureCode.EXEC_FAIL in code:
                suggestions.append("Check SQL syntax and table/column names")
                suggestions.append("Verify Snowflake connection and permissions")
            
            elif FailureCode.SEMANTIC_DRIFT in code:
                suggestions.append("Ensure SQL maintains original semantic intent")
                suggestions.append("Check that key entities are preserved")
            
            elif FailureCode.KF_MISMATCH in code:
                suggestions.append("Adjust Altair code to satisfy key features")
                suggestions.append("Check encoding mappings match requirements")
            
            elif FailureCode.CONTRACT_VIOLATION in code:
                suggestions.append("Ensure chart type contract is met")
                suggestions.append("Consider falling back to simpler chart type")
            
            elif FailureCode.INTERACT_DEGENERATE in code:
                suggestions.append("Ensure selection affects visible elements")
                suggestions.append("Check that filter/condition uses selection")
            
            elif FailureCode.AMB_WEAK in code:
                suggestions.append("Remove explicit mentions of ambiguous features")
                suggestions.append("Use more vague language")
            
            elif FailureCode.AMB_STRONG in code:
                suggestions.append("Add more context to question")
                suggestions.append("Reduce number of ambiguous points")
        
        return list(set(suggestions))


# Singleton instance
_validator: Optional[Validator] = None


def get_validator() -> Validator:
    """Get or create singleton validator."""
    global _validator
    if _validator is None:
        _validator = Validator()
    return _validator
