"""
Ambiguity Injector for Vis-Interact Dataset Construction.

Post-Altair ambiguity injection with diversity control.
Injects ambiguity AFTER Altair code is successfully generated,
using the actual vega_lite_spec to determine valid injection points.

Ambiguity Types (based on Benchmark-Design.md):
- DATA: Aggregation, time granularity, filter scope ambiguity
- VISUALIZATION: Chart type, encoding choices ambiguity  
- INFO_COMPLETION: Missing info needed to connect vis with data
- ERROR_CORRECTION: Conflicts with data facts or incompatible vis/data

This module includes the ambiguity template library (merged from ambiguity_planner).
"""
import json
import random
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.models import (
    AmbiguityProfile, AmbiguityType, PreferredInterface, Difficulty,
    FeatureCandidate, SemanticContext, RequiredKeyFeature, RequiredKeyFeatureType,
)
from utils.llm_client import get_llm_client
from utils.diversity import get_diversity_tracker

logger = logging.getLogger(__name__)


# =========================================================================
# Ambiguity Template Library (merged from ambiguity_planner)
# =========================================================================

@dataclass
class AmbiguityTemplate:
    ambiguity_type: AmbiguityType
    name: str
    description: str
    target_paths: List[str]
    preferred_interface: PreferredInterface
    typical_difficulty: Difficulty
    injection_patterns: List[str]


AMBIGUITY_TEMPLATES: Dict[str, AmbiguityTemplate] = {
    # =========================================================================
    # DATA Ambiguity
    # - Aggregation method unclear
    # - Time granularity missing
    # - Top-K not specified
    # =========================================================================
    "aggregation_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.DATA,
        name="Aggregation Missing",
        description="Aggregation method (sum/avg/count/median) not specified",
        target_paths=["encoding.y.aggregate", "encoding.x.aggregate"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.EASY,
        injection_patterns=[
            "overall", "in general", "typical", "values",
            "total performance", "how it looks"
        ],
    ),
    "time_granularity_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.DATA,
        name="Time Granularity Missing",
        description="Time granularity (day/week/month/year) not specified",
        target_paths=["encoding.x.timeUnit", "encoding.y.timeUnit"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.EASY,
        injection_patterns=[
            "over time", "trend", "across the period", "historically",
            "temporal pattern", "time series"
        ],
    ),
    "topk_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.DATA,
        name="Top-K Missing",
        description="Number of items (Top-K) not specified",
        target_paths=["transform.filter", "transform.window"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.EASY,
        injection_patterns=[
            "top items", "leading categories", "main ones", "key players",
            "most important", "highest ranking"
        ],
    ),
    "filter_scope_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.DATA,
        name="Filter Scope Missing",
        description="Data filter/scope not clearly specified (which subset of data)",
        target_paths=["transform.filter"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.EASY,
        injection_patterns=[
            "recent data", "relevant entries", "applicable records",
            "related data", "corresponding records"
        ],
    ),

    # =========================================================================
    # VISUALIZATION Ambiguity
    # - Chart type not specified
    # - Encoding channel unclear
    # - Layout/composition unspecified
    # =========================================================================
    "chart_type_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.VISUALIZATION,
        name="Chart Type Missing",
        description="Specific chart type not mentioned",
        target_paths=["mark.type"],
        preferred_interface=PreferredInterface.VIS,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "visualize", "show graphically", "display", "plot",
            "create a chart", "draw a graph"
        ],
    ),
    "encoding_channel_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.VISUALIZATION,
        name="Encoding Channel Missing",
        description="Color/size/shape encoding field not specified",
        target_paths=["encoding.color.field", "encoding.size.field", "encoding.shape.field"],
        preferred_interface=PreferredInterface.VIS,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "distinguish categories", "highlight differences", "differentiate",
            "separate by type", "show distinctions"
        ],
    ),
    "axis_assignment_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.VISUALIZATION,
        name="Axis Assignment Missing",
        description="Which field goes on which axis not specified",
        target_paths=["encoding.x.field", "encoding.y.field"],
        preferred_interface=PreferredInterface.VIS,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "compare", "relationship between", "correlation",
            "how they relate", "contrast"
        ],
    ),

    # =========================================================================
    # INFO_COMPLETION
    # - Missing dimension for aggregation
    # - Missing filter conditions
    # - Missing grouping criteria
    # =========================================================================
    "bucket_definition_needed": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.INFO_COMPLETION,
        name="Bucket Definition Needed",
        description="Need to define bins/buckets for continuous data",
        target_paths=["encoding.x.bin", "transform.bin"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "distribution", "spread", "range breakdown",
            "how values are distributed", "value ranges"
        ],
    ),
    "grouping_dimension_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.INFO_COMPLETION,
        name="Grouping Dimension Missing",
        description="Pie/donut chart needs grouping dimension specified",
        target_paths=["encoding.theta.field", "encoding.color.field"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "breakdown", "composition", "proportion",
            "share", "makeup", "structure"
        ],
    ),
    "comparison_baseline_missing": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.INFO_COMPLETION,
        name="Comparison Baseline Missing",
        description="Comparison needs a baseline or reference point",
        target_paths=["transform.calculate", "encoding.y.datum"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "compare to", "versus", "against",
            "relative to", "benchmarked"
        ],
    ),

    # =========================================================================
    # ERROR_CORRECTION
    # - Time range exceeds data coverage
    # - Field name doesn't exist in schema
    # - Chart type incompatible with data
    # =========================================================================
    "data_range_mismatch": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.ERROR_CORRECTION,
        name="Data Range Mismatch",
        description="Time range or value range exceeds data coverage",
        target_paths=["transform.filter"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.MEDIUM,
        injection_patterns=[
            "in 2025", "all-time", "complete history", "since beginning",
            "from the start", "entire period"
        ],
    ),
    "field_name_error": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.ERROR_CORRECTION,
        name="Field Name Error",
        description="Field name doesn't exist in schema (typo or wrong name)",
        target_paths=["encoding.*.field"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.HARD,
        injection_patterns=[],  # Dynamic based on schema - populated at injection time
    ),
    "vis_data_incompatible": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.ERROR_CORRECTION,
        name="Vis-Data Incompatible",
        description="Chart type incompatible with data nature (e.g., pie chart for continuous data)",
        target_paths=["mark.type"],
        preferred_interface=PreferredInterface.VIS,
        typical_difficulty=Difficulty.HARD,
        injection_patterns=[
            "pie chart for continuous", "line for categorical",
            "bar for temporal sequence", "scatter for single value"
        ],
    ),
    "aggregation_conflict": AmbiguityTemplate(
        ambiguity_type=AmbiguityType.ERROR_CORRECTION,
        name="Aggregation Conflict",
        description="Requested aggregation conflicts with data (e.g., avg of categorical)",
        target_paths=["encoding.y.aggregate"],
        preferred_interface=PreferredInterface.TEXT,
        typical_difficulty=Difficulty.HARD,
        injection_patterns=[
            "average of names", "sum of categories",
            "median of labels", "count unique numbers"
        ],
    ),
}


class AmbiguityPlanner:
    """Planner for ambiguity injection guidance."""
    
    def __init__(self):
        self.templates = AMBIGUITY_TEMPLATES

    def get_injection_guidance(self, profile: AmbiguityProfile, original_text: str) -> Dict[str, Any]:
        """Return guidance used by the NL rewriting stage."""
        guidance: Dict[str, Any] = {
            "omit_info": [],
            "vague_phrases": [],
            "error_to_inject": profile.corruption,
        }

        for _, template in self.templates.items():
            # Check if template's type is in profile's types list
            if template.ambiguity_type not in profile.ambiguity_types:
                continue
            if any(p in profile.target_feature_paths for p in template.target_paths):
                guidance["omit_info"].append(template.description)
                guidance["vague_phrases"].extend(template.injection_patterns)

        # A tiny shuffle to avoid always picking the same phrases
        random.shuffle(guidance["vague_phrases"])
        return guidance


_planner: Optional[AmbiguityPlanner] = None


def get_ambiguity_planner() -> AmbiguityPlanner:
    """Get or create singleton ambiguity planner."""
    global _planner
    if _planner is None:
        _planner = AmbiguityPlanner()
    return _planner


# =========================================================================
# Ambiguity Injector
# =========================================================================

@dataclass
class AmbiguityDiversityConstraints:
    """Constraints to guide ambiguity type selection for diversity."""
    preferred_types: List[str] = None  # Underrepresented ambiguity types
    avoid_types: List[str] = None      # Overrepresented types to avoid
    preferred_templates: List[str] = None  # Underrepresented templates
    avoid_templates: List[str] = None  # Overrepresented templates
    strength: float = 0.5  # 0.0 = suggestion, 1.0 = requirement
    
    def __post_init__(self):
        self.preferred_types = self.preferred_types or []
        self.avoid_types = self.avoid_types or []
        self.preferred_templates = self.preferred_templates or []
        self.avoid_templates = self.avoid_templates or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_types": self.preferred_types,
            "avoid_types": self.avoid_types,
            "preferred_templates": self.preferred_templates,
            "avoid_templates": self.avoid_templates,
            "strength": self.strength
        }


class AmbiguityInjector:
    """
    Post-Altair ambiguity injection with diversity control.
    
    Injects ambiguity AFTER Altair code is successfully generated,
    using the actual vega_lite_spec to determine valid injection points.
    
    Key Features:
    - Uses vega_spec to identify valid ambiguity injection points
    - Supports diversity constraints to balance ambiguity type distribution
    - Provides multiple ambiguity templates per type
    """
    
    # Template names grouped by ambiguity type
    TYPE_TO_TEMPLATES = {
        AmbiguityType.DATA: [
            "aggregation_missing",
            "time_granularity_missing",
            "topk_missing",
            "filter_scope_missing",
        ],
        AmbiguityType.VISUALIZATION: [
            "chart_type_missing",
            "encoding_channel_missing",
            "axis_assignment_missing",
        ],
        AmbiguityType.INFO_COMPLETION: [
            "bucket_definition_needed",
            "grouping_dimension_missing",
            "comparison_baseline_missing",
        ],
        AmbiguityType.ERROR_CORRECTION: [
            "data_range_mismatch",
            "field_name_error",
            "vis_data_incompatible",
            "aggregation_conflict",
        ],
    }
    
    def __init__(self):
        self.llm = get_llm_client()
        self.ambiguity_planner = get_ambiguity_planner()
    
    def inject(
        self,
        clear_question: str,
        candidate: FeatureCandidate,
        vega_spec: Dict[str, Any],
        semantic_ctx: SemanticContext,
        diversity_constraints: Optional[AmbiguityDiversityConstraints] = None,
        required_key_features: Optional[List[RequiredKeyFeature]] = None
    ) -> Tuple[str, AmbiguityProfile]:
        """
        Inject ambiguity into a clear question.
        
        Uses vega_spec to identify valid ambiguity injection points
        (e.g., actual encodings, transforms, mark type).
        
        Args:
            clear_question: The clear, unambiguous question
            candidate: The visualization candidate (FeatureCandidate)
            vega_spec: The generated Vega-Lite specification
            semantic_ctx: Semantic context with schema info
            diversity_constraints: Constraints for diversity control
            required_key_features: NL-based required features for guiding injection
                                   - must=True features should not be corrupted
                                   - must=False features can be made ambiguous
            
        Returns:
            Tuple of (ambiguous_question, ambiguity_profile)
        """
        # 1. Analyze vega_spec to find valid injection points
        # Pass required_key_features to filter out templates that would corrupt must=True features
        valid_templates = self._find_valid_templates(
            vega_spec, candidate, semantic_ctx, required_key_features
        )
        
        if not valid_templates:
            logger.warning("No valid ambiguity templates found, using fallback")
            valid_templates = ["chart_type_missing"]  # Fallback
        
        # 2. Select top candidate templates based on actual usage distribution
        # Get diversity tracker to access usage statistics
        tracker = get_diversity_tracker()
        candidate_templates = self._select_top_templates_with_diversity(
            valid_templates, tracker, top_k=3, diversity_constraints=diversity_constraints
        )
        
        # 3. Generate ambiguous question via LLM (now returns list of templates)
        ambiguous_question, selected_template_names = self._rewrite_question(
            clear_question, candidate_templates, candidate, vega_spec
        )
        
        # 4. Build ambiguity profile using ALL selected templates
        if not selected_template_names:
            # Fallback if no templates selected
            logger.warning("No templates selected by LLM, using first candidate template")
            selected_template_names = [candidate_templates[0]]
        
        selected_types = []
        all_target_paths = []
        all_interfaces = set()
        all_difficulties = []
        
        for template_name in selected_template_names:
            template = AMBIGUITY_TEMPLATES.get(template_name)
            if template:
                if template.ambiguity_type not in selected_types:
                    selected_types.append(template.ambiguity_type)
                all_target_paths.extend(template.target_paths)
                all_interfaces.add(template.preferred_interface)
                all_difficulties.append(template.typical_difficulty)
            else:
                logger.warning(f"Template {template_name} not found in AMBIGUITY_TEMPLATES")
        
        # Fallback if no valid templates found
        if not selected_types:
            logger.warning("No valid templates found, using fallback")
            fallback_template = AMBIGUITY_TEMPLATES.get(candidate_templates[0])
            if fallback_template:
                selected_types = [fallback_template.ambiguity_type]
                all_target_paths = fallback_template.target_paths
                all_interfaces = {fallback_template.preferred_interface}
                all_difficulties = [fallback_template.typical_difficulty]
                selected_template_names = [candidate_templates[0]]
        
        # Use most common/highest difficulty as representative
        profile = AmbiguityProfile(
            ambiguity_types=selected_types,
            template_names=selected_template_names,
            target_feature_paths=list(set(all_target_paths)),
            preferred_interface=list(all_interfaces)[0] if all_interfaces else PreferredInterface.TEXT,
            difficulty=max(all_difficulties) if all_difficulties else Difficulty.EASY
        )
        
        # DEBUG: Log the injection result
        logger.info(f"🔍 Ambiguity injection:")
        logger.info(f"  Selected {len(selected_template_names)} template(s): {selected_template_names}")
        logger.info(f"  Types: {[t.value for t in selected_types]}")
        logger.info(f"  Clear: {clear_question[:80]}...")
        logger.info(f"  Ambiguous: {ambiguous_question[:80]}...")
        logger.info(f"  Same?: {clear_question == ambiguous_question}")
        
        return ambiguous_question, profile
    
    def _find_valid_templates(
        self,
        vega_spec: Dict[str, Any],
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        required_key_features: Optional[List[RequiredKeyFeature]] = None
    ) -> List[str]:
        """
        Analyze vega_spec to find templates that can be validly applied.
        
        Args:
            vega_spec: The Vega-Lite specification
            candidate: The FeatureCandidate
            semantic_ctx: Semantic context
            required_key_features: NL-based features (kept for compatibility but not used for filtering)
        
        Returns:
            List of template names that are compatible with the spec.
        """
        valid = []
        
        # Check encoding-based templates
        encoding = vega_spec.get("encoding", {})
        
        # DATA templates
        # aggregation_missing: if there's an aggregate in y encoding
        if encoding.get("y", {}).get("aggregate"):
            valid.append("aggregation_missing")
        
        # time_granularity_missing: if there's a timeUnit or temporal field
        if (encoding.get("x", {}).get("timeUnit") or 
            encoding.get("x", {}).get("type") == "temporal"):
            valid.append("time_granularity_missing")
        
        # topk_missing: if there's a transform with filter or window
        transforms = vega_spec.get("transform", [])
        if any("filter" in t or "window" in t for t in transforms):
            valid.append("topk_missing")
        
        # filter_scope_missing: if there are filters in transforms
        if any("filter" in t for t in transforms):
            valid.append("filter_scope_missing")
        
        # VISUALIZATION templates
        # chart_type_missing: always valid as user can be vague about chart type
        valid.append("chart_type_missing")
        
        # encoding_channel_missing: if there's a color/size/shape encoding
        if any(ch in encoding for ch in ["color", "size", "shape"]):
            valid.append("encoding_channel_missing")
        
        # axis_assignment_missing: if there are x and y encodings
        if "x" in encoding and "y" in encoding:
            valid.append("axis_assignment_missing")
        
        # INFO_COMPLETION templates
        # bucket_definition_needed: if there's binning
        if any(enc.get("bin") for enc in encoding.values() if isinstance(enc, dict)):
            valid.append("bucket_definition_needed")
        
        # grouping_dimension_missing: for pie/donut charts
        mark = vega_spec.get("mark", {})
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        if mark_type == "arc":
            valid.append("grouping_dimension_missing")
        
        # comparison_baseline_missing: if there are calculated fields
        if any("calculate" in t for t in transforms):
            valid.append("comparison_baseline_missing")
        
        # ERROR_CORRECTION templates
        # data_range_mismatch: can always be added
        valid.append("data_range_mismatch")

        # vis_data_incompatible: targets mark.type
        valid.append("vis_data_incompatible")

        # field_name_error: targets encoding.*.field
        # Only meaningful if there is at least one encoding field in the spec
        if any(
            isinstance(enc, dict) and enc.get("field")
            for enc in encoding.values()
        ):
            valid.append("field_name_error")

        # aggregation_conflict: targets encoding.y.aggregate
        # Meaningful when we have a y encoding (measure exists)
        if isinstance(encoding.get("y"), dict) and (encoding["y"].get("field") or encoding["y"].get("aggregate")):
            valid.append("aggregation_conflict")
        
        return valid
    
    def _select_top_templates_with_diversity(
        self,
        valid_templates: List[str],
        tracker,
        top_k: int = 3,
        diversity_constraints: Optional[AmbiguityDiversityConstraints] = None
    ) -> List[str]:
        """
        Select top K candidate templates based on actual usage distribution.
        
        Uses inverse frequency weighting: templates used less frequently get higher weights.
        This automatically balances the distribution over time without manual thresholds.
        
        Args:
            valid_templates: List of template names that are valid for current spec
            tracker: DiversityTracker instance with usage statistics
            top_k: Number of templates to return
            
        Returns:
            List of template names, prioritizing least-used templates
        """
        if not valid_templates:
            return []
        
        # Get current template usage statistics from tracker
        template_usage = {}
        if tracker and hasattr(tracker, 'feature_space'):
            # Access ambiguity_templates counter from feature_space
            ambiguity_templates_count = getattr(tracker.feature_space, 'ambiguity_templates', {})
            for template_name in valid_templates:
                template_usage[template_name] = ambiguity_templates_count.get(template_name, 0)
        else:
            # No statistics available, all templates get equal weight
            for template_name in valid_templates:
                template_usage[template_name] = 0
        
        # Calculate weights based on inverse frequency + diversity constraints
        # Less used templates get higher weights, and constraints can boost/penalize.
        weighted_templates: List[Tuple[str, float]] = []

        strength = float(getattr(diversity_constraints, "strength", 0.5) or 0.5) if diversity_constraints else 0.5
        preferred_templates = set(getattr(diversity_constraints, "preferred_templates", []) or []) if diversity_constraints else set()
        avoid_templates = set(getattr(diversity_constraints, "avoid_templates", []) or []) if diversity_constraints else set()
        preferred_types = set(getattr(diversity_constraints, "preferred_types", []) or []) if diversity_constraints else set()
        avoid_types = set(getattr(diversity_constraints, "avoid_types", []) or []) if diversity_constraints else set()

        # FILTER STEP: In early stage or with high strength, strictly filter out avoided templates
        # This ensures diversity in parallel processing when multiple candidates are generated simultaneously
        should_filter = (strength > 0.5 or len(template_usage) < 10)  # Early stage or strong constraints
        
        for template_name in valid_templates:
            # Strict filtering for avoided templates/types (NEW)
            if should_filter:
                template = AMBIGUITY_TEMPLATES.get(template_name)
                # Skip templates that are explicitly avoided
                if template_name in avoid_templates:
                    continue
                # Skip templates whose type is avoided
                if template and template.ambiguity_type:
                    type_value = template.ambiguity_type.value if hasattr(template.ambiguity_type, "value") else str(template.ambiguity_type)
                    if type_value in avoid_types:
                        continue
            
            usage_count = template_usage.get(template_name, 0)
            # Inverse frequency: weight = 1 / (count + 1)
            base = 1.0 / (usage_count + 1)

            # Template-level boosts/penalties
            if template_name in preferred_templates:
                base *= (1.0 + 1.5 * strength)
            if template_name in avoid_templates:
                base *= max(0.01, 1.0 - 0.9 * strength)

            # Type-level boosts/penalties
            template = AMBIGUITY_TEMPLATES.get(template_name)
            if template and template.ambiguity_type:
                type_value = template.ambiguity_type.value if hasattr(template.ambiguity_type, "value") else str(template.ambiguity_type)
                if type_value in preferred_types:
                    base *= (1.0 + 1.0 * strength)
                if type_value in avoid_types:
                    base *= max(0.01, 1.0 - 0.9 * strength)

            # Break ties deterministically but non-statically across runs
            # (Python sort is stable; without jitter equal weights always keep original order)
            # Increased perturbation from 1e-6 to 0.1 to effectively break ties
            base += random.random() * 0.1

            weighted_templates.append((template_name, base))
        
        # Use weighted random sampling instead of simple top-K selection
        # This ensures diversity even when weights are similar
        if len(weighted_templates) <= top_k:
            # If we have fewer templates than requested, return all
            top_templates = [name for name, _ in weighted_templates]
        else:
            # Normalize weights to probabilities
            total_weight = sum(weight for _, weight in weighted_templates)
            if total_weight > 0:
                # Sample with replacement, then deduplicate
                names = [name for name, _ in weighted_templates]
                weights = [weight for _, weight in weighted_templates]
                # Sample top_k * 2 to ensure we get enough unique templates
                sampled = random.choices(names, weights=weights, k=min(top_k * 2, len(names)))
                # Deduplicate while preserving order
                seen = set()
                top_templates = []
                for name in sampled:
                    if name not in seen:
                        seen.add(name)
                        top_templates.append(name)
                        if len(top_templates) >= top_k:
                            break
                # If we still don't have enough, fill with highest weighted remaining
                if len(top_templates) < top_k:
                    remaining = [(name, weight) for name, weight in weighted_templates if name not in seen]
                    remaining.sort(key=lambda x: x[1], reverse=True)
                    top_templates.extend([name for name, _ in remaining[:top_k - len(top_templates)]])
            else:
                # Fallback to simple top-K if all weights are zero
                weighted_templates.sort(key=lambda x: x[1], reverse=True)
                top_templates = [name for name, _ in weighted_templates[:top_k]]
        
        return top_templates
    
    def _rewrite_question(
        self,
        clear_question: str,
        candidate_template_names: List[str],
        candidate: FeatureCandidate,
        vega_spec: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Use LLM to rewrite clear question with injected ambiguity.
        
        Returns:
            Tuple of (ambiguous_question, selected_template_name)
        """
        # Build info for all candidate templates
        template_options = []
        for template_name in candidate_template_names:
            template = AMBIGUITY_TEMPLATES.get(template_name)
            if not template:
                continue
            
            vague_phrases = ", ".join(template.injection_patterns[:3]) if template.injection_patterns else "N/A"
            target_info = self._describe_target(template, vega_spec)
            
            # IMPORTANT: Expose the internal template key so the model can return it exactly.
            template_options.append({
                'name': template.name,
                'key': template_name,
                'type': template.ambiguity_type.value,
                'description': template.description,
                'target_info': target_info,
                'vague_phrases': vague_phrases
            })
        
        # Shuffle options to avoid LLM always selecting the first one
        random.shuffle(template_options)
        
        # Format options with renumbered Option 1, 2, 3...
        template_options_formatted = []
        for idx, opt in enumerate(template_options):
            template_options_formatted.append(f"""
### Option {idx + 1}: {opt['name']} (`{opt['key']}`)
- **Type**: {opt['type']}
- **Description**: {opt['description']}
- **What to make ambiguous**: {opt['target_info']}
- **Natural phrases**: {opt['vague_phrases']}
""")
        
        # Get type guidance for all represented types
        type_guidances = []
        seen_types = set()
        for template_name in candidate_template_names:
            template = AMBIGUITY_TEMPLATES.get(template_name)
            if template and template.ambiguity_type not in seen_types:
                seen_types.add(template.ambiguity_type)
                type_guidances.append(self._get_type_guidance(template.ambiguity_type))
        
        template_options_text = "\n".join(template_options_formatted)
        type_guidance_text = "\n".join(type_guidances)
        
        prompt = f"""Rewrite this visualization question to simulate how a NON-TECHNICAL user (like a business manager or client) would naturally express their visualization needs.

## Current Clear Question (technical, complete)
{clear_question}

## User Persona
You are simulating a **non-technical user** who:
- Doesn't know exact data terminology (aggregation functions, time units, etc.)
- Focuses on business insights, not technical details
- May be vague about visualization preferences
- May have incomplete information or minor misconceptions about the data
- Speaks naturally, like asking a colleague for help
- **Often includes redundant or irrelevant information**:
  * Background context that doesn't affect the visualization
  * Repeated phrases or ideas
  * Unnecessary details about why they need it
  * Conversational fillers ("I was thinking...", "maybe we could...")
- **May ramble a bit** before getting to the point

## Ambiguity Options (you may select MULTIPLE options)
{template_options_text}

## Selection Strategy
You can select ONE or MULTIPLE ambiguity templates:
- **Single selection**: Choose the most natural single ambiguity point
- **Multiple selection**: Combine 2-3 templates if they naturally coexist in a user question
  * Can combine across types (e.g., DATA + VISUALIZATION)
  * Can combine within same type (e.g., aggregation_missing + time_granularity_missing)
  * Keep it realistic - don't force unnatural combinations

## Type-Specific Guidelines
{type_guidance_text}

## Rewriting Guidelines
1. **Choose the most natural option(s)** from the ambiguity options above:
   - Select ONE template that fits most naturally with the question context
   - OR combine 2-3 templates if they work well together naturally
   - Prioritize what feels most realistic for a non-technical user to ask

2. **Sound like a real person**, not a technical specification:
   - ✓ "Can you show me how sales are doing over time?"
   - ✗ "Generate a line chart of monthly aggregated sales."

3. **Add natural redundancy and filler** (realistic user behavior):
   - Include unnecessary context: "I'm preparing for the meeting and I need to..."
   - Add conversational fillers: "I was thinking maybe...", "It would be great if..."
   - Repeat the request in different words: "Show me the data... I want to see how it looks"
   - Include irrelevant details: "My boss asked about this yesterday, can you..."
   - Be slightly verbose: 2-4 sentences instead of 1 clean sentence
   
4. **For DATA ambiguity**: Use vague business terms
   - Instead of "SUM/AVG/COUNT", say "total", "overall", "how many"
   - Instead of "monthly/yearly", say "over time", "trends"

5. **For VISUALIZATION ambiguity**: Don't specify chart types
   - Instead of "bar chart", say "show", "visualize", "compare"
   - Focus on WHAT insight is needed, not HOW to display it

6. **For INFO_COMPLETION**: Leave out critical details naturally
   - E.g., "Show me the distribution" (but don't say which bins/ranges)
   - E.g., "Compare the performance" (but don't say to what baseline)

7. **For ERROR_CORRECTION**: Include subtle errors a non-tech user might make
   - Mention a time range that may exceed data coverage
   - Use slightly wrong field names (e.g., "customer name" vs "client name")
   - Request incompatible vis+data combinations without realizing it

8. **Keep it natural and conversational**:
   - 2-4 sentences (a bit verbose, like real users)
   - Use "I want", "Can you", "Show me", "I need to" style phrasing
   - Should feel like a Slack message or email request
   - It's OK to be a bit rambling or indirect

## Output Format
Return a JSON object with:
```json
{{
    "rewritten_question": "the ambiguous question from user's perspective",
    "selected_templates": ["template_key_1", "template_key_2"],
    "reasoning": "brief explanation of why these templates were selected together (if multiple)"
}}
```

Note: `selected_templates` should be a list of template keys, even if only one is selected.
"""
        
        response = self.llm.complete(
            prompt,
            response_format={"type": "json_object"},
            process_name="ambiguity_injector.rewrite_question"
        )
        
        def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
            """
            Best-effort extraction of a JSON object from an LLM response.
            Handles:
            - raw JSON: {"rewritten_question": "...", "selected_template": "..."}
            - fenced JSON blocks: ```json { ... } ```
            - extra pre/post text around the JSON object
            """
            if not text:
                return None
            s = text.strip()

            # 1) fenced code block
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except Exception:
                    pass

            # 2) raw JSON
            try:
                return json.loads(s)
            except Exception:
                pass

            # 3) first {...} span
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start : end + 1])
                except Exception:
                    return None

            return None

        if response and response.content:
            raw = response.content.strip()
            result = _extract_json_object(raw)
            rewritten = str(result.get("rewritten_question", "")).strip()
            
            # NEW: Handle both single and multiple template selection
            selected = result.get("selected_templates", result.get("selected_template", []))
            if isinstance(selected, str):
                selected = [selected]  # Convert single string to list
            elif not isinstance(selected, list):
                selected = []
            
            # Filter out empty strings
            selected = [s for s in selected if s]
            
            return rewritten, selected  # Now returns list of templates
        else:
            rewritten, selected = "", []
            logger.warning(f"Failed to rewrite question: {response.content}")
        
        return rewritten, selected
    
    def _get_type_guidance(self, ambiguity_type: AmbiguityType) -> str:
        """Get specific guidance for each ambiguity type."""
        guidance_map = {
            AmbiguityType.DATA: """
**DATA Ambiguity** - Non-technical users often don't know exact data operations:
- Say "total" instead of specifying SUM/AVG/COUNT/MEDIAN
- Say "over time" instead of daily/weekly/monthly/yearly
- Say "recent data" or "top items" without exact filters
- Use business language: "performance", "trends", "patterns", not aggregation functions
- **Add redundancy**: Include background context, repeat ideas, be verbose
- Example (concise): "Show me sales performance"
- Example (with redundancy): "Hey, I'm working on the quarterly report and I need to understand our sales performance. Can you show me how we're doing? I want to see the overall numbers and trends."
""",
            AmbiguityType.VISUALIZATION: """
**VISUALIZATION Ambiguity** - Non-technical users focus on insights, not chart types:
- Say "show", "compare", "visualize" without mentioning bar/line/pie charts
- Don't specify encodings like "X-axis should be...", "use color for..."
- Focus on business goal: "I want to compare regions" not "make a bar chart"
- Let the system decide HOW to visualize, user only says WHAT to understand
- **Add redundancy**: Ramble about the purpose, repeat the request
- Example (concise): "I need to see how different categories compare"
- Example (with redundancy): "So I've been thinking about this for a while, and I really need to understand how our different categories are performing. Can you help me visualize this data? I just want to be able to compare them easily and see which ones are doing better."
""",
            AmbiguityType.INFO_COMPLETION: """
**INFO_COMPLETION** - User's request lacks key details for vis+data alignment:
- Request visualization type that needs more specification for data mapping
  * "Show distribution" → but which bins/ranges?
  * "Make a pie chart" → but pie of what? (need grouping dimension)
  * "Compare performance" → but to what baseline?
- The visualization TYPE is mentioned, but HOW it maps to DATA is unclear
- System must ask to complete the missing semantic connection
- **Add redundancy**: Include reasoning about why they want it, but still miss key details
- Example (concise): "Use a pie chart to show sales"
- Example (with redundancy): "For tomorrow's presentation, I think a pie chart would look really good to show our sales data. My manager likes pie charts because they're easy to understand. Can you make one for me? I need it to look professional."
""",
            AmbiguityType.ERROR_CORRECTION: """
**ERROR_CORRECTION** - User makes mistakes about data or incompatible requests:
- **Data fact errors**: Wrong time range ("show 2025 data" when data ends in 2023)
- **Schema errors**: Wrong field names ("customer_name" when it's actually "client_name")
- **Vis+Data incompatibility**: Request chart type that doesn't fit data nature
  * Pie chart for continuous time series
  * Line chart for categorical comparisons without ordering
- These are ERRORS the system should catch and correct, not just ambiguities
- **Add redundancy**: Confidently make the wrong request with extra context
- Example (concise): "Show me a pie chart of daily temperature trends"
- Example (with redundancy): "I was looking at last year's weather report and I think it would be really helpful to see the temperature trends. I'm thinking a pie chart would be perfect for this because it's easy to read. Can you create a pie chart showing how the temperature changed day by day over the past year?"
"""
        }
        return guidance_map.get(ambiguity_type, "")
    
    def _describe_target(self, template: AmbiguityTemplate, vega_spec: Dict[str, Any]) -> str:
        """Describe what specifically should be made ambiguous."""
        descriptions = {
            "aggregation_missing": "The aggregation function (SUM, AVG, COUNT, etc.) - remove or vague-ify",
            "time_granularity_missing": "The time unit (daily, monthly, yearly) - make it vague like 'over time'",
            "topk_missing": "The number of items (Top-5, Top-10) - say 'top' without specifying how many",
            "filter_scope_missing": "The data filter/scope - be vague about which subset of data",
            "chart_type_missing": "Don't mention any specific chart type - just ask to 'visualize' or 'show'",
            "encoding_channel_missing": "The encoding choice (color, size) - just ask to 'differentiate' or 'distinguish'",
            "axis_assignment_missing": "Which field goes where - just ask to 'compare' or show 'relationship'",
            "bucket_definition_needed": "The bin/bucket size for distribution - just ask for 'distribution'",
            "grouping_dimension_missing": "What to group by for the pie/composition - just ask for 'breakdown'",
            "comparison_baseline_missing": "The comparison baseline/reference - just ask to 'compare' without specifying to what",
            "data_range_mismatch": "Mention a time range that may be outside data coverage",
            "field_name_error": "Use a slightly wrong or ambiguous field name",
            "vis_data_incompatible": "Request a visualization that may not fit the data type",
            "aggregation_conflict": "Request an aggregation that may not make sense for the data type",
        }
        return descriptions.get(template.name, template.description)


# Singleton instance
_injector: Optional[AmbiguityInjector] = None


def get_ambiguity_injector() -> AmbiguityInjector:
    """Get or create singleton ambiguity injector."""
    global _injector
    if _injector is None:
        _injector = AmbiguityInjector()
    return _injector
