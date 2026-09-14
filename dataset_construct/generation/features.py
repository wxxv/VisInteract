"""
Required Key Feature Generator for Vis-Interact Dataset Construction.

Generates RequiredKeyFeature (NL-based) from vega_spec + candidate.
These represent non-technical user's "hard requirements" for metric evaluation.

The RequiredKeyFeature format follows Metric.md:
- id: stable identifier (e.g., "kf_mark_bar")
- type: feature category (mark/encoding/aggregation/filter/overlay_stat_line/interaction/composition)
- text: NL description (core): "Use a bar chart."
- args: optional structured params: {"mark": "bar"}
- must: if True, cannot be corrupted by ambiguity injection
"""
import json
import logging
from typing import Any, Dict, List, Optional

from core.models import (
    RequiredKeyFeature, RequiredKeyFeatureType, FeatureCandidate, SemanticContext
)
from utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class RequiredKeyFeatureGenerator:
    """
    Generates RequiredKeyFeature (NL-based) from vega_spec + candidate.
    
    Uses LLM to analyze the visualization and generate NL descriptions
    that simulate what a non-technical user would specify as requirements.
    """
    
    # Mapping from string type names to enum values
    TYPE_MAPPING = {
        "mark": RequiredKeyFeatureType.MARK,
        "encoding": RequiredKeyFeatureType.ENCODING,
        "aggregation": RequiredKeyFeatureType.AGGREGATION,
        "filter": RequiredKeyFeatureType.FILTER,
        "overlay_stat_line": RequiredKeyFeatureType.OVERLAY_STAT_LINE,
        "interaction": RequiredKeyFeatureType.INTERACTION,
        "composition": RequiredKeyFeatureType.COMPOSITION,
    }
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def generate(
        self,
        vega_spec: Dict[str, Any],
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        min_features: int = 2,
        max_features: int = 5
    ) -> List[RequiredKeyFeature]:
        """
        Generate RequiredKeyFeature list from vega_spec and candidate.
        
        Uses LLM to analyze the visualization and generate NL descriptions
        that simulate what a non-technical user would specify as requirements.
        
        Args:
            vega_spec: The generated Vega-Lite specification
            candidate: The FeatureCandidate that produced this visualization
            semantic_ctx: Semantic context with schema and domain info
            min_features: Minimum number of features to generate
            max_features: Maximum number of features to generate
        
        Returns:
            List of RequiredKeyFeature objects
        """
        try:
            # Use LLM to generate features
            llm_features = self._llm_generate_features(
                vega_spec, candidate, semantic_ctx, min_features, max_features
            )
            
            if llm_features:
                return llm_features
            
            # Fallback to heuristic generation if LLM fails
            logger.warning("LLM feature generation failed, using heuristic fallback")
            return self._heuristic_generate_features(vega_spec, candidate)
            
        except Exception as e:
            logger.error(f"Error generating required key features: {e}")
            return self._heuristic_generate_features(vega_spec, candidate)
    
    def _llm_generate_features(
        self,
        vega_spec: Dict[str, Any],
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        min_features: int,
        max_features: int
    ) -> List[RequiredKeyFeature]:
        """Use LLM to generate NL-based required key features."""
        
        # Build context for LLM
        # Truncate datasets field to avoid excessive token usage
        spec_for_prompt = self._truncate_datasets(vega_spec, max_items=3)
        spec_json = json.dumps(spec_for_prompt, indent=2, ensure_ascii=False)
        
        prompt = f"""You are simulating a NON-TECHNICAL business user who is requesting a data visualization.

## Task
Analyze this generated visualization and extract the KEY REQUIREMENTS that a non-technical user would naturally specify when requesting this chart.

## Visualization Spec (Vega-Lite)
```json
{spec_json}
```

## Business Context
- Analytical Question: {candidate.analytical_question or "Not specified"}
- Visualization Intent: {candidate.vis_intent or "Not specified"}
- Chart Type: {candidate.chart_type}

## Guidelines
Extract {min_features}-{max_features} key requirements that a NON-TECHNICAL user would naturally express:

1. **Focus on WHAT users want to see, not HOW it's implemented**
   - ✓ "I want to see a bar chart" (not "mark.type should be bar")
   - ✓ "Show sales by month" (not "x encoding should use temporal field with month timeUnit")

2. **Use natural, business-oriented language**
   - ✓ "Compare the total revenue across different regions"
   - ✓ "Show me a line showing the average over time"
   - ✗ "aggregate: sum on encoding.y.field"

3. **Identify key visual elements a user would notice/request**
   - Chart type (bar, line, pie, etc.)
   - Key data being shown (which fields, what aggregation in plain terms)
   - Special visual elements (reference lines, color coding, interactions)

4. **Mark MUST features vs optional**
   - must=true: Core requirements the user explicitly requested (e.g., chart type, specific data)
   - must=false: Features that could vary without breaking user's intent

## Required Key Feature Types
- **mark**: Chart type (bar/line/scatter/pie/area)
- **encoding**: Data fields and their visual mapping
- **aggregation**: How data is summarized (total, average, count, etc.)
- **filter**: Data scope/filtering
- **overlay_stat_line**: Reference lines (average line, target line)
- **interaction**: Interactive features (filters, brushing, tooltips)
- **composition**: Multi-view or layered structure

## Output Format
Return a JSON object with "required_key_features" array:
```json
{{
    "required_key_features": [
        {{
            "id": "kf_mark_bar",
            "type": "mark",
            "text": "Use a bar chart to display the data.",
            "args": {{"mark": "bar"}},
            "must": true
        }},
        {{
            "id": "kf_agg_sum",
            "type": "aggregation",
            "text": "Show the total sales amount.",
            "args": {{"aggregate": "sum", "field": "sales"}},
            "must": true
        }}
    ]
}}
```

Output only the JSON object.
"""
        
        result = self.llm.complete_json(
            prompt,
            process_name="required_key_feature_generator.generate"
        )
        
        if not result or "required_key_features" not in result:
            return []
        
        features = []
        for i, item in enumerate(result["required_key_features"]):
            try:
                # Parse type
                type_str = item.get("type", "mark").lower()
                feature_type = self.TYPE_MAPPING.get(type_str, RequiredKeyFeatureType.MARK)
                
                # Create feature
                feature = RequiredKeyFeature(
                    id=item.get("id", f"kf_{i}"),
                    type=feature_type,
                    text=item.get("text", ""),
                    args=item.get("args", {}),
                    must=item.get("must", True)
                )
                
                # Validate - must have non-empty text
                if feature.text.strip():
                    features.append(feature)
                    
            except Exception as e:
                logger.debug(f"Failed to parse feature {i}: {e}")
                continue
        
        return features
    
    def _heuristic_generate_features(
        self,
        vega_spec: Dict[str, Any],
        candidate: FeatureCandidate
    ) -> List[RequiredKeyFeature]:
        """
        Generate features using heuristics when LLM fails.
        Extracts key information directly from vega_spec.
        """
        features = []
        feature_idx = 0
        
        # 1. Mark type (always include)
        mark = vega_spec.get("mark", {})
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        if mark_type:
            mark_text = self._mark_to_nl(mark_type)
            features.append(RequiredKeyFeature(
                id=f"kf_mark_{mark_type}",
                type=RequiredKeyFeatureType.MARK,
                text=mark_text,
                args={"mark": mark_type},
                must=True
            ))
            feature_idx += 1
        
        # 2. Key encoding (x/y)
        encoding = vega_spec.get("encoding", {})
        
        # X encoding
        if "x" in encoding:
            x_enc = encoding["x"]
            x_field = x_enc.get("field", "")
            x_timeunit = x_enc.get("timeUnit", "")
            if x_field:
                if x_timeunit:
                    text = f"Show data by {x_field} ({x_timeunit})."
                else:
                    text = f"Organize data by {x_field}."
                features.append(RequiredKeyFeature(
                    id=f"kf_enc_x_{feature_idx}",
                    type=RequiredKeyFeatureType.ENCODING,
                    text=text,
                    args={"channel": "x", "field": x_field, "timeUnit": x_timeunit},
                    must=True
                ))
                feature_idx += 1
        
        # Y encoding with aggregation
        if "y" in encoding:
            y_enc = encoding["y"]
            y_field = y_enc.get("field", "")
            y_agg = y_enc.get("aggregate", "")
            if y_field and y_agg:
                agg_nl = self._aggregate_to_nl(y_agg)
                text = f"Show the {agg_nl} of {y_field}."
                features.append(RequiredKeyFeature(
                    id=f"kf_agg_{y_agg}_{feature_idx}",
                    type=RequiredKeyFeatureType.AGGREGATION,
                    text=text,
                    args={"aggregate": y_agg, "field": y_field},
                    must=True
                ))
                feature_idx += 1
        
        # 3. Color encoding (if present)
        if "color" in encoding:
            color_enc = encoding["color"]
            color_field = color_enc.get("field", "")
            if color_field:
                text = f"Color-code by {color_field}."
                features.append(RequiredKeyFeature(
                    id=f"kf_enc_color_{feature_idx}",
                    type=RequiredKeyFeatureType.ENCODING,
                    text=text,
                    args={"channel": "color", "field": color_field},
                    must=False  # Color coding is often optional
                ))
                feature_idx += 1
        
        # 4. Check for layers (stat lines, etc.)
        if "layer" in vega_spec:
            for layer in vega_spec["layer"]:
                layer_mark = layer.get("mark", {})
                layer_mark_type = layer_mark.get("type") if isinstance(layer_mark, dict) else layer_mark
                
                # Check for rule (reference line)
                if layer_mark_type == "rule":
                    layer_enc = layer.get("encoding", {})
                    y_enc = layer_enc.get("y", {})
                    y_agg = y_enc.get("aggregate", "")
                    if y_agg in ["mean", "median"]:
                        text = f"Add a reference line showing the {y_agg}."
                        features.append(RequiredKeyFeature(
                            id=f"kf_stat_line_{y_agg}_{feature_idx}",
                            type=RequiredKeyFeatureType.OVERLAY_STAT_LINE,
                            text=text,
                            args={"stat": y_agg, "orientation": "horizontal"},
                            must=True
                        ))
                        feature_idx += 1
        
        # 5. Check for transforms (filters, etc.)
        transforms = vega_spec.get("transform", [])
        for transform in transforms:
            if "filter" in transform:
                filter_expr = transform["filter"]
                if isinstance(filter_expr, str):
                    text = f"Filter the data: {filter_expr}."
                    features.append(RequiredKeyFeature(
                        id=f"kf_filter_{feature_idx}",
                        type=RequiredKeyFeatureType.FILTER,
                        text=text,
                        args={"filter": filter_expr},
                        must=False
                    ))
                    feature_idx += 1
        
        # 6. Check for selection/interaction
        if "params" in vega_spec or "selection" in vega_spec:
            features.append(RequiredKeyFeature(
                id=f"kf_interaction_{feature_idx}",
                type=RequiredKeyFeatureType.INTERACTION,
                text="Make the chart interactive.",
                args={},
                must=False
            ))
            feature_idx += 1
        
        return features
    
    def _mark_to_nl(self, mark_type: str) -> str:
        """Convert mark type to natural language description."""
        mark_nl_map = {
            "bar": "Use a bar chart.",
            "line": "Use a line chart.",
            "point": "Use a scatter plot.",
            "area": "Use an area chart.",
            "arc": "Use a pie/donut chart.",
            "rect": "Use a heatmap.",
            "rule": "Use a reference line.",
            "boxplot": "Use a box plot.",
            "circle": "Use a bubble chart.",
            "square": "Use a square marker chart.",
            "tick": "Use a tick mark chart.",
            "text": "Show text labels.",
            "geoshape": "Use a geographic map.",
        }
        return mark_nl_map.get(mark_type, f"Use a {mark_type} chart.")
    
    def _aggregate_to_nl(self, agg: str) -> str:
        """Convert aggregate function to natural language."""
        agg_nl_map = {
            "sum": "total",
            "mean": "average",
            "average": "average",
            "count": "count",
            "min": "minimum",
            "max": "maximum",
            "median": "median",
            "distinct": "unique count",
            "variance": "variance",
            "stdev": "standard deviation",
        }
        return agg_nl_map.get(agg.lower(), agg)
    
    def _truncate_datasets(self, vega_spec: Dict[str, Any], max_items: int = 3) -> Dict[str, Any]:
        """
        Truncate datasets field in Vega-Lite spec to reduce token usage.
        Shows distinct counts for each field to help LLM understand data distribution.
        
        Args:
            vega_spec: Original Vega-Lite specification
            max_items: Maximum number of data items to keep per dataset
            
        Returns:
            Modified spec with truncated datasets and field statistics
        """
        import copy
        
        # Create a deep copy to avoid modifying the original
        spec_copy = copy.deepcopy(vega_spec)
        
        # Check if datasets field exists
        if "datasets" not in spec_copy:
            return spec_copy
        
        datasets = spec_copy["datasets"]
        if not isinstance(datasets, dict):
            return spec_copy
        
        # Truncate each dataset
        for dataset_name, data_items in datasets.items():
            if isinstance(data_items, list) and len(data_items) > max_items:
                total_count = len(data_items)
                remaining_count = total_count - max_items
                
                # Calculate distinct counts for each field
                field_distinct_counts = {}
                if data_items:
                    # Get all fields from the first item
                    sample_item = data_items[0]
                    if isinstance(sample_item, dict):
                        for field in sample_item.keys():
                            # Count distinct values for this field
                            distinct_values = set()
                            for item in data_items:
                                if isinstance(item, dict) and field in item:
                                    value = item[field]
                                    # Handle different types
                                    if value is not None:
                                        # Convert to string for hashing
                                        distinct_values.add(str(value))
                            field_distinct_counts[field] = len(distinct_values)
                
                # Format field statistics
                field_stats = ", ".join([f"{field}: {count} distinct" for field, count in sorted(field_distinct_counts.items())])
                
                # Keep first max_items and add a placeholder for remaining
                truncated_items = data_items[:max_items]
                truncated_items.append({
                    "_truncated": f"... and {remaining_count} more items (total: {total_count} items)",
                    "_field_statistics": field_stats
                })
                
                spec_copy["datasets"][dataset_name] = truncated_items
        
        return spec_copy


# Singleton instance
_generator: Optional[RequiredKeyFeatureGenerator] = None


def get_required_key_feature_generator() -> RequiredKeyFeatureGenerator:
    """Get or create singleton required key feature generator."""
    global _generator
    if _generator is None:
        _generator = RequiredKeyFeatureGenerator()
    return _generator
