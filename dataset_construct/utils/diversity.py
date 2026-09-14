"""
Diversity Tracker for Key Feature diversity control.
Tracks and manages diversity across multiple dimensions of key features.
"""
import logging
import math
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.models import VisSample, Feature

logger = logging.getLogger(__name__)


# All possible values for each dimension
AGGREGATION_TYPES = ["sum", "avg", "count", "max", "min", "median", "distinct", "variance", "stdev"]
TIME_UNITS = ["year", "quarter", "month", "week", "day", "hour", "minute", "yearmonth", "monthdate"]
TRANSFORM_TYPES = ["aggregate", "window", "bin", "calculate", "filter", "fold", "pivot", "density", "loess", "regression"]
COMPOSITION_TYPES = ["single", "layer", "hconcat", "vconcat", "facet", "repeat"]
INTERACTION_TYPES = ["none", "brush", "slider", "legend", "point_selection", "crossfilter"]
ENCODING_TYPES = ["Q", "N", "O", "T"]  # Quantitative, Nominal, Ordinal, Temporal

# Ambiguity types and templates for diversity tracking
AMBIGUITY_TYPES = ["DATA", "VISUALIZATION", "INFO_COMPLETION", "ERROR_CORRECTION"]
AMBIGUITY_TEMPLATES = [
    # DATA
    "aggregation_missing", "time_granularity_missing", "topk_missing", "filter_scope_missing",
    # VISUALIZATION
    "chart_type_missing", "encoding_channel_missing", "axis_assignment_missing",
    # INFO_COMPLETION
    "bucket_definition_needed", "grouping_dimension_missing", "comparison_baseline_missing",
    # ERROR_CORRECTION
    "data_range_mismatch", "field_name_error", "vis_data_incompatible", "aggregation_conflict"
]

# Domain keywords for inferring data domain
DOMAIN_KEYWORDS = {
    "finance": ["revenue", "sales", "price", "stock", "trading", "bank", "transaction", "payment"],
    "healthcare": ["patient", "medical", "health", "hospital", "drug", "disease", "clinical"],
    "retail": ["product", "customer", "order", "cart", "shop", "store", "inventory"],
    "analytics": ["event", "session", "user", "click", "page", "visitor", "traffic"],
    "geographic": ["location", "region", "country", "city", "geo", "map", "latitude", "longitude"],
    "temporal": ["date", "time", "year", "month", "day", "period", "duration"],
    "scientific": ["experiment", "sample", "measurement", "observation", "sensor"],
    "social": ["user", "follower", "post", "comment", "like", "share", "network"],
}


@dataclass
class ReservationRecord:
    """记录一次预占操作"""
    reservation_id: str
    reserved_chart_types: List[str] = field(default_factory=list)
    reserved_aggregations: List[str] = field(default_factory=list)
    reserved_time_units: List[str] = field(default_factory=list)
    reserved_transforms: List[str] = field(default_factory=list)
    reserved_compositions: List[str] = field(default_factory=list)
    reserved_interactions: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class FeatureSpace:
    """Tracks distribution of key features across multiple dimensions."""
    encoding_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    aggregations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    time_units: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    transforms: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    compositions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    interactions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    domains: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Chart type and category tracking
    chart_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    chart_categories: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Ambiguity tracking
    ambiguity_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ambiguity_templates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Track field usage for encoding diversity
    x_field_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    y_field_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    color_field_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "encoding_patterns": dict(self.encoding_patterns),
            "aggregations": dict(self.aggregations),
            "time_units": dict(self.time_units),
            "transforms": dict(self.transforms),
            "compositions": dict(self.compositions),
            "interactions": dict(self.interactions),
            "domains": dict(self.domains),
            "chart_types": dict(self.chart_types),
            "chart_categories": dict(self.chart_categories),
            "ambiguity_types": dict(self.ambiguity_types),
            "ambiguity_templates": dict(self.ambiguity_templates),
        }


@dataclass
class DiversityConstraints:
    """Constraints to guide key feature generation for diversity."""
    preferred_aggregations: List[str] = field(default_factory=list)
    preferred_time_units: List[str] = field(default_factory=list)
    preferred_encoding_pattern: Optional[str] = None
    preferred_transforms: List[str] = field(default_factory=list)
    preferred_composition: Optional[str] = None
    preferred_interaction: Optional[str] = None
    
    avoid_aggregations: List[str] = field(default_factory=list)
    avoid_time_units: List[str] = field(default_factory=list)
    avoid_encoding_patterns: List[str] = field(default_factory=list)
    
    # Chart type diversity (new)
    preferred_chart_types: List[str] = field(default_factory=list)
    preferred_chart_categories: List[str] = field(default_factory=list)
    avoid_chart_types: List[str] = field(default_factory=list)
    
    # Strength of constraint (0.0 = suggestion, 1.0 = requirement)
    strength: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_aggregations": self.preferred_aggregations,
            "preferred_time_units": self.preferred_time_units,
            "preferred_encoding_pattern": self.preferred_encoding_pattern,
            "preferred_transforms": self.preferred_transforms,
            "preferred_composition": self.preferred_composition,
            "preferred_interaction": self.preferred_interaction,
            "avoid_aggregations": self.avoid_aggregations,
            "avoid_time_units": self.avoid_time_units,
            "avoid_encoding_patterns": self.avoid_encoding_patterns,
            "preferred_chart_types": self.preferred_chart_types,
            "preferred_chart_categories": self.preferred_chart_categories,
            "avoid_chart_types": self.avoid_chart_types,
            "strength": self.strength,
        }


class DiversityTracker:
    """Tracks and manages diversity of key features across the dataset."""
    
    def __init__(self):
        self.feature_space = FeatureSpace()
        self.total_samples = 0
        self._sample_ids: Set[str] = set()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Reservation tracking for parallel processing
        self._reservations: Dict[str, ReservationRecord] = {}
    
    def record_sample(self, sample: VisSample):
        """
        Record a sample's features to update diversity tracking.
        Thread-safe version with lock protection.
        
        Supports both legacy Sample and new VisSample models.
        """
        with self._lock:
            # Get sample_id (compatible with both models)
            sample_id = getattr(sample, 'sample_id', None) or getattr(sample, 'case_id', None)
            if not sample_id:
                return
            
            if sample_id in self._sample_ids:
                return  # Already recorded
            
            self._sample_ids.add(sample_id)
            self.total_samples += 1
        
        # Get key features
        key_features = sample.key_features or []
        
        # Get vega spec (VisSample uses vega_lite_spec, Sample uses gt_spec)
        spec = getattr(sample, 'vega_lite_spec', None) or getattr(sample, 'gt_spec', None) or {}
        
        # Get db_id (VisSample has db_id directly, Sample has database_info.db_id)
        db_id = getattr(sample, 'db_id', None)
        if not db_id:
            database_info = getattr(sample, 'database_info', None)
            db_id = database_info.db_id if database_info else ""
        
        # Get chart type and category
        chart_type = getattr(sample, 'chart_type', None) or ""
        # VisSample may not have chart_category, derive from chart_type if needed
        chart_category = getattr(sample, 'chart_category', None) or self._infer_chart_category(chart_type)
        
        # Extract and record features
        self._record_encoding_pattern(key_features)
        self._record_aggregations(key_features)
        self._record_time_units(key_features)
        self._record_transforms(key_features, spec)
        self._record_composition(spec)
        self._record_interaction(key_features, spec)
        self._record_domain(db_id)
        self._record_chart_type(chart_type, chart_category)
        
        # Record ambiguity info
        ambiguity_profile = getattr(sample, 'ambiguity_profile', None)
        if ambiguity_profile:
            self._record_ambiguity(ambiguity_profile)
    
    def _record_chart_type(self, chart_type: str, chart_category: str):
        """Record chart type and category usage."""
        if chart_type:
            self.feature_space.chart_types[chart_type] += 1
        if chart_category:
            self.feature_space.chart_categories[chart_category] += 1
    
    def _record_ambiguity(self, profile):
        """Record ambiguity type and template usage."""
        # Record all types
        for amb_type in profile.ambiguity_types:
            type_value = amb_type.value if hasattr(amb_type, 'value') else str(amb_type)
            self.feature_space.ambiguity_types[type_value] += 1
        
        # Record all templates
        for template_name in profile.template_names:
            if template_name:  # Skip empty strings
                self.feature_space.ambiguity_templates[template_name] += 1
    
    def _infer_chart_category(self, chart_type: str) -> str:
        """Infer chart category from chart type name."""
        if not chart_type:
            return ""
        
        chart_type_lower = chart_type.lower()
        
        # Map chart types to categories
        if any(x in chart_type_lower for x in ["bar", "column"]):
            return "Bar Charts"
        elif any(x in chart_type_lower for x in ["line", "area", "stream"]):
            return "Line/Area Charts"
        elif any(x in chart_type_lower for x in ["scatter", "point", "bubble"]):
            return "Scatter/Point Charts"
        elif any(x in chart_type_lower for x in ["pie", "donut", "arc"]):
            return "Pie/Arc Charts"
        elif any(x in chart_type_lower for x in ["histogram", "density"]):
            return "Distribution Charts"
        elif any(x in chart_type_lower for x in ["box", "violin"]):
            return "Statistical Charts"
        elif any(x in chart_type_lower for x in ["heat", "matrix"]):
            return "Heatmaps"
        elif any(x in chart_type_lower for x in ["geo", "map", "choropleth"]):
            return "Geographic Charts"
        else:
            return "Other"
    
    def record_chart_type(self, chart_type: str, chart_category: str):
        """Public method to record chart type (for use outside of sample recording)."""
        self._record_chart_type(chart_type, chart_category)
    
    def get_chart_type_coverage_gaps(self, all_chart_types: List[str]) -> List[str]:
        """Get chart types with zero coverage."""
        return [ct for ct in all_chart_types if self.feature_space.chart_types.get(ct, 0) == 0]
    
    def get_underrepresented_chart_types(
        self, 
        all_chart_types: List[str],
        threshold: float = 0.5
    ) -> List[str]:
        """Get chart types with below-average coverage."""
        if not self.feature_space.chart_types:
            return all_chart_types
        
        total = sum(self.feature_space.chart_types.values())
        expected = total / len(all_chart_types) if all_chart_types else 0
        
        return [
            ct for ct in all_chart_types
            if self.feature_space.chart_types.get(ct, 0) < expected * threshold
        ]
    
    def get_overrepresented_chart_types(self, threshold: float = 2.0) -> List[str]:
        """Get chart types with above-average coverage."""
        if not self.feature_space.chart_types:
            return []
        
        total = sum(self.feature_space.chart_types.values())
        expected = total / len(self.feature_space.chart_types) if self.feature_space.chart_types else 0
        
        # Special handling for early stage: if total samples is small,
        # consider any chart_type with count > 0 as already represented
        # This prevents over-selection of the same chart_type during parallel instance processing
        if total < 10:
            # In early stage, avoid any chart_type that already has samples
            # unless it's significantly underrepresented compared to others
            return [ct for ct, count in self.feature_space.chart_types.items() if count > 0]
        
        return [
            ct for ct, count in self.feature_space.chart_types.items()
            if count > expected * threshold
        ]
    
    def _record_encoding_pattern(self, key_features: List[Feature]):
        """Extract and record encoding pattern (e.g., 'T-Q-N')."""
        pattern = self.extract_encoding_pattern(key_features)
        if pattern:
            self.feature_space.encoding_patterns[pattern] += 1
        
        # Also record individual channel types
        for feature in key_features:
            if feature.path == "encoding.x.type" and feature.value:
                self.feature_space.x_field_types[str(feature.value)] += 1
            elif feature.path == "encoding.y.type" and feature.value:
                self.feature_space.y_field_types[str(feature.value)] += 1
            elif feature.path == "encoding.color.type" and feature.value:
                self.feature_space.color_field_types[str(feature.value)] += 1
    
    def _record_aggregations(self, key_features: List[Feature]):
        """Record aggregation types used."""
        for feature in key_features:
            if "aggregate" in feature.path.lower() and feature.value:
                agg = str(feature.value).lower()
                self.feature_space.aggregations[agg] += 1
    
    def _record_time_units(self, key_features: List[Feature]):
        """Record time units used."""
        for feature in key_features:
            if "timeUnit" in feature.path and feature.value:
                tu = str(feature.value).lower()
                self.feature_space.time_units[tu] += 1
    
    def _record_transforms(self, key_features: List[Feature], spec: Dict[str, Any]):
        """Record transform types used."""
        # From key features
        for feature in key_features:
            if feature.path.startswith("transform"):
                for tt in TRANSFORM_TYPES:
                    if tt in feature.path.lower():
                        self.feature_space.transforms[tt] += 1
                        break
        
        # From spec
        transforms = spec.get("transform", [])
        if transforms:
            for t in transforms:
                for tt in TRANSFORM_TYPES:
                    if tt in t:
                        self.feature_space.transforms[tt] += 1
    
    def _record_composition(self, spec: Dict[str, Any]):
        """Record composition type used."""
        if "layer" in spec:
            self.feature_space.compositions["layer"] += 1
        elif "hconcat" in spec:
            self.feature_space.compositions["hconcat"] += 1
        elif "vconcat" in spec:
            self.feature_space.compositions["vconcat"] += 1
        elif "facet" in spec:
            self.feature_space.compositions["facet"] += 1
        elif "repeat" in spec:
            self.feature_space.compositions["repeat"] += 1
        else:
            self.feature_space.compositions["single"] += 1
    
    def _record_interaction(self, key_features: List[Feature], spec: Dict[str, Any]):
        """Record interaction type used."""
        has_interaction = False
        
        # Check params in spec
        params = spec.get("params", [])
        for param in params:
            if "select" in param:
                select = param["select"]
                if isinstance(select, dict):
                    select_type = select.get("type", "")
                    if select_type == "interval":
                        self.feature_space.interactions["brush"] += 1
                        has_interaction = True
                    elif select_type == "point":
                        self.feature_space.interactions["point_selection"] += 1
                        has_interaction = True
            if "bind" in param:
                bind = param["bind"]
                if bind == "legend":
                    self.feature_space.interactions["legend"] += 1
                    has_interaction = True
                elif isinstance(bind, dict) or bind == "scales":
                    self.feature_space.interactions["slider"] += 1
                    has_interaction = True
        
        # Check key features for interaction hints
        for feature in key_features:
            if "params" in feature.path or "select" in feature.path:
                has_interaction = True
        
        if not has_interaction:
            self.feature_space.interactions["none"] += 1
    
    def _record_domain(self, db_id: str):
        """Infer and record data domain from database ID."""
        db_lower = db_id.lower()
        domain_found = False
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in db_lower for kw in keywords):
                self.feature_space.domains[domain] += 1
                domain_found = True
                break
        
        if not domain_found:
            self.feature_space.domains["other"] += 1
    
    @staticmethod
    def extract_encoding_pattern(key_features: List[Feature]) -> str:
        """Extract encoding type pattern from key features."""
        pattern_parts = []
        
        for channel in ['x', 'y', 'color', 'size']:
            type_feature = next(
                (f for f in key_features if f.path == f"encoding.{channel}.type"),
                None
            )
            if type_feature and type_feature.value:
                # Get first letter (Q/N/O/T)
                value = str(type_feature.value)
                if value.lower().startswith("quant"):
                    pattern_parts.append("Q")
                elif value.lower().startswith("nomin"):
                    pattern_parts.append("N")
                elif value.lower().startswith("ordin"):
                    pattern_parts.append("O")
                elif value.lower().startswith("temp"):
                    pattern_parts.append("T")
                else:
                    pattern_parts.append(value[0].upper())
        
        return "-".join(pattern_parts) if pattern_parts else ""
    
    def get_underrepresented(
        self,
        dimension: str,
        threshold: float = 0.5,
        all_values: Optional[List[str]] = None
    ) -> List[str]:
        """Get values that are underrepresented in a dimension."""
        counts = getattr(self.feature_space, dimension, {})
        
        if not counts and all_values:
            # Nothing recorded yet, return all as underrepresented
            return all_values
        
        if not counts:
            return []
        
        # Calculate expected count for uniform distribution
        total = sum(counts.values())
        num_categories = len(all_values) if all_values else len(counts)
        if num_categories == 0:
            return []
        
        expected = total / num_categories
        
        # Find underrepresented values
        underrepresented = []
        
        if all_values:
            for value in all_values:
                count = counts.get(value, 0)
                if count < expected * threshold:
                    underrepresented.append(value)
        else:
            for value, count in counts.items():
                if count < expected * threshold:
                    underrepresented.append(value)
        
        return underrepresented
    
    def get_overrepresented(
        self,
        dimension: str,
        threshold: float = 1.5
    ) -> List[str]:
        """Get values that are overrepresented in a dimension."""
        counts = getattr(self.feature_space, dimension, {})
        
        if not counts:
            return []
        
        total = sum(counts.values())
        expected = total / len(counts) if counts else 0
        
        # Special handling for early stage to enforce diversity in parallel processing
        # For ambiguity types and templates, use stricter criteria in early stage
        if dimension in ["ambiguity_types", "ambiguity_templates"] and total < 10:
            # In early stage, any item with count > 0 should be avoided to maximize diversity
            return [k for k, v in counts.items() if v > 0]
        
        return [k for k, v in counts.items() if v > expected * threshold]
    
    def calculate_diversity_score(self, dimension: str) -> float:
        """Calculate diversity score (normalized entropy) for a dimension."""
        counts = getattr(self.feature_space, dimension, {})
        
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by maximum entropy
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def get_overall_diversity_score(self) -> float:
        """Calculate overall diversity score across all dimensions."""
        dimensions = [
            "encoding_patterns", "aggregations", "time_units",
            "transforms", "compositions", "interactions", "domains"
        ]
        
        scores = [self.calculate_diversity_score(dim) for dim in dimensions]
        return sum(scores) / len(scores) if scores else 0.0
    
    def reserve_recommendations(
        self,
        chart_type: Optional[str] = None,
        chart_category: Optional[str] = None,
        all_chart_types: Optional[List[str]] = None
    ) -> Tuple[str, 'DiversityConstraints']:
        """
        Get diversity recommendations and reserve all recommended options optimistically.
        Thread-safe: use lock to protect state read and write.
        
        Args:
            chart_type: Optional chart type hint
            chart_category: Optional chart category hint
            all_chart_types: List of all available chart types
            
        Returns:
            (reservation_id, diversity_constraints)
        """
        with self._lock:
            # 1. Generate recommendations (based on current state)
            constraints = self.get_diversity_constraints(
                chart_type, chart_category, None, all_chart_types
            )
            
            # 2. Reserve all recommended options (+1)
            reservation_id = str(uuid.uuid4())
            
            # Reserve chart_types (first 3)
            reserved_chart_types = constraints.preferred_chart_types[:3] if constraints.preferred_chart_types else []
            for ct in reserved_chart_types:
                self.feature_space.chart_types[ct] += 1
            
            # Reserve aggregations
            reserved_aggregations = constraints.preferred_aggregations[:] if constraints.preferred_aggregations else []
            for agg in reserved_aggregations:
                self.feature_space.aggregations[agg] += 1
            
            # Reserve time_units
            reserved_time_units = constraints.preferred_time_units[:] if constraints.preferred_time_units else []
            for tu in reserved_time_units:
                self.feature_space.time_units[tu] += 1
            
            # Reserve transforms
            reserved_transforms = constraints.preferred_transforms[:] if constraints.preferred_transforms else []
            for tf in reserved_transforms:
                self.feature_space.transforms[tf] += 1
            
            # Reserve composition (only one)
            reserved_compositions = [constraints.preferred_composition] if constraints.preferred_composition else []
            for comp in reserved_compositions:
                self.feature_space.compositions[comp] += 1
            
            # Reserve interaction (only one)
            reserved_interactions = [constraints.preferred_interaction] if constraints.preferred_interaction else []
            for inter in reserved_interactions:
                self.feature_space.interactions[inter] += 1
            
            # Record reservation information
            self._reservations[reservation_id] = ReservationRecord(
                reservation_id=reservation_id,
                reserved_chart_types=reserved_chart_types,
                reserved_aggregations=reserved_aggregations,
                reserved_time_units=reserved_time_units,
                reserved_transforms=reserved_transforms,
                reserved_compositions=reserved_compositions,
                reserved_interactions=reserved_interactions,
                timestamp=time.time()
            )
            
            logger.debug(f"Reserved {reservation_id}: {len(reserved_chart_types)} chart_types, "
                        f"{len(reserved_aggregations)} aggregations")
            
            return reservation_id, constraints
    
    def commit_selection(
        self,
        reservation_id: str,
        sample: Optional[VisSample],
        skip_rollback: bool = False
    ):
        """
        提交实际选中的options，回退未选中的预占。
        
        Args:
            reservation_id: ID returned by reserve_recommendations
            sample: Generated successful sample (None means generation failed)
            skip_rollback: If True, only clean up reservation records without rolling back count
                          (used for cases where record_sample has already recorded the actual sample)
        """
        with self._lock:
            if reservation_id not in self._reservations:
                logger.warning(f"Unknown reservation_id: {reservation_id}")
                return
            
            record = self._reservations[reservation_id]
            
            if skip_rollback:
                # Only clean up reservation records, do not roll back count
                # Suitable for cases where record_sample() has already recorded the actual sample
                logger.debug(f"Cleaning reservation {reservation_id} without rollback (samples already recorded)")
            elif sample is None:
                # Generation failed, rollback all reservations
                self._rollback_reservation(record)
                logger.debug(f"Rolled back reservation {reservation_id} (generation failed)")
            else:
                # Generation successful, only rollback unselected
                self._commit_with_rollback(record, sample)
                logger.debug(f"Committed reservation {reservation_id} with chart_type={sample.chart_type}")
            
            # Clean up reservation records
            del self._reservations[reservation_id]
    
    def _rollback_reservation(self, record: ReservationRecord):
        """Rollback reservation: all reservations -1 (no lock, held by caller)"""
        for chart_type in record.reserved_chart_types:
            if chart_type in self.feature_space.chart_types:
                self.feature_space.chart_types[chart_type] -= 1
                # Avoid negative numbers
                if self.feature_space.chart_types[chart_type] < 0:
                    self.feature_space.chart_types[chart_type] = 0
        
        for agg in record.reserved_aggregations:
            if agg in self.feature_space.aggregations:
                self.feature_space.aggregations[agg] -= 1
                if self.feature_space.aggregations[agg] < 0:
                    self.feature_space.aggregations[agg] = 0
        
        for tu in record.reserved_time_units:
            if tu in self.feature_space.time_units:
                self.feature_space.time_units[tu] -= 1
                if self.feature_space.time_units[tu] < 0:
                    self.feature_space.time_units[tu] = 0
        
        for tf in record.reserved_transforms:
            if tf in self.feature_space.transforms:
                self.feature_space.transforms[tf] -= 1
                if self.feature_space.transforms[tf] < 0:
                    self.feature_space.transforms[tf] = 0
        
        for comp in record.reserved_compositions:
            if comp in self.feature_space.compositions:
                self.feature_space.compositions[comp] -= 1
                if self.feature_space.compositions[comp] < 0:
                    self.feature_space.compositions[comp] = 0
        
        for inter in record.reserved_interactions:
            if inter in self.feature_space.interactions:
                self.feature_space.interactions[inter] -= 1
                if self.feature_space.interactions[inter] < 0:
                    self.feature_space.interactions[inter] = 0
    
    def _commit_with_rollback(self, record: ReservationRecord, sample: VisSample):
        """Commit selected, rollback unselected (no lock, held by caller)"""
        # Get actual used features from sample
        selected_chart = sample.chart_type
        
        # Rollback unselected chart_types
        for chart_type in record.reserved_chart_types:
            if chart_type != selected_chart:
                if chart_type in self.feature_space.chart_types:
                    self.feature_space.chart_types[chart_type] -= 1
                    if self.feature_space.chart_types[chart_type] < 0:
                        self.feature_space.chart_types[chart_type] = 0
        
        # For aggregations, time_units, etc., extract actual used from sample's key_features
        key_features = sample.key_features or []
        
        # Extract actual used aggregations
        used_aggregations = set()
        for feature in key_features:
            if "aggregate" in feature.path.lower() and feature.value:
                used_aggregations.add(str(feature.value).lower())
        
        # Rollback unused aggregations
        for agg in record.reserved_aggregations:
            if agg not in used_aggregations:
                if agg in self.feature_space.aggregations:
                    self.feature_space.aggregations[agg] -= 1
                    if self.feature_space.aggregations[agg] < 0:
                        self.feature_space.aggregations[agg] = 0
        
        # Extract actual used time_units
        used_time_units = set()
        for feature in key_features:
            if "timeUnit" in feature.path and feature.value:
                used_time_units.add(str(feature.value).lower())
        
        # Rollback unused time_units
        for tu in record.reserved_time_units:
            if tu not in used_time_units:
                if tu in self.feature_space.time_units:
                    self.feature_space.time_units[tu] -= 1
                    if self.feature_space.time_units[tu] < 0:
                        self.feature_space.time_units[tu] = 0
        
        # For transforms, compositions, interactions, simply rollback all (because it's difficult to extract exactly from features)
        # Note: record_sample will correctly +1 the actual used value
        for tf in record.reserved_transforms:
            if tf in self.feature_space.transforms:
                self.feature_space.transforms[tf] -= 1
                if self.feature_space.transforms[tf] < 0:
                    self.feature_space.transforms[tf] = 0
        
        for comp in record.reserved_compositions:
            if comp in self.feature_space.compositions:
                self.feature_space.compositions[comp] -= 1
                if self.feature_space.compositions[comp] < 0:
                    self.feature_space.compositions[comp] = 0
        
        for inter in record.reserved_interactions:
            if inter in self.feature_space.interactions:
                self.feature_space.interactions[inter] -= 1
                if self.feature_space.interactions[inter] < 0:
                    self.feature_space.interactions[inter] = 0
    
    def get_diversity_constraints(
        self,
        chart_type: Optional[str] = None,
        chart_category: Optional[str] = None,
        db_domain: Optional[str] = None,
        all_chart_types: Optional[List[str]] = None
    ) -> DiversityConstraints:
        """Generate diversity constraints for the next sample."""
        constraints = DiversityConstraints()
        
        # Determine strength based on how many samples we have
        if self.total_samples < 10:
            constraints.strength = 0.3  # Weak constraints early on
        elif self.total_samples < 50:
            constraints.strength = 0.5
        else:
            constraints.strength = 0.7  # Stronger constraints as we accumulate
        
        # Find underrepresented aggregations
        underrep_aggs = self.get_underrepresented(
            "aggregations", threshold=0.5, all_values=AGGREGATION_TYPES
        )
        if underrep_aggs:
            constraints.preferred_aggregations = underrep_aggs[:3]
        
        # Find overrepresented aggregations to avoid
        overrep_aggs = self.get_overrepresented("aggregations", threshold=2.0)
        constraints.avoid_aggregations = overrep_aggs
        
        # Find underrepresented time units
        underrep_tu = self.get_underrepresented(
            "time_units", threshold=0.5, all_values=TIME_UNITS
        )
        if underrep_tu:
            constraints.preferred_time_units = underrep_tu[:3]
        
        # Find overrepresented time units to avoid
        overrep_tu = self.get_overrepresented("time_units", threshold=2.0)
        constraints.avoid_time_units = overrep_tu
        
        # Find underrepresented transforms
        underrep_transforms = self.get_underrepresented(
            "transforms", threshold=0.5, all_values=TRANSFORM_TYPES
        )
        if underrep_transforms:
            constraints.preferred_transforms = underrep_transforms[:3]
        
        # Find underrepresented compositions
        underrep_comp = self.get_underrepresented(
            "compositions", threshold=0.5, all_values=COMPOSITION_TYPES
        )
        if underrep_comp:
            constraints.preferred_composition = underrep_comp[0]
        
        # Find underrepresented interactions
        underrep_inter = self.get_underrepresented(
            "interactions", threshold=0.5, all_values=INTERACTION_TYPES
        )
        if underrep_inter:
            # Filter out "none" as a preferred option unless chart supports interaction
            inter_options = [i for i in underrep_inter if i != "none"]
            if inter_options and chart_category in ["Interactive Charts"]:
                constraints.preferred_interaction = inter_options[0]
        
        # Find underrepresented encoding patterns
        underrep_patterns = self.get_underrepresented("encoding_patterns", threshold=0.3)
        overrep_patterns = self.get_overrepresented("encoding_patterns", threshold=2.0)
        
        if underrep_patterns:
            constraints.preferred_encoding_pattern = underrep_patterns[0]
        constraints.avoid_encoding_patterns = overrep_patterns
        
        # Chart type diversity constraints
        if all_chart_types:
            underrep_chart_types = self.get_underrepresented_chart_types(all_chart_types)
            if underrep_chart_types:
                # Provide more chart types for richer diversity guidance
                constraints.preferred_chart_types = underrep_chart_types[:10]
            
            overrep_chart_types = self.get_overrepresented_chart_types()
            constraints.avoid_chart_types = overrep_chart_types
        
        # Chart category diversity constraints
        underrep_categories = self.get_underrepresented(
            "chart_categories", threshold=0.5
        )
        if underrep_categories:
            constraints.preferred_chart_categories = underrep_categories[:5]
        
        return constraints
    
    def get_ambiguity_constraints(self) -> Dict[str, Any]:
        """
        Get constraints to guide ambiguity injection for diversity.
        
        Returns a dict with:
        - preferred_types: Underrepresented ambiguity types
        - avoid_types: Overrepresented types to avoid
        - preferred_templates: Underrepresented templates
        - avoid_templates: Overrepresented templates
        """
        # Find underrepresented ambiguity types
        underrep_types = self.get_underrepresented(
            "ambiguity_types", threshold=0.5, all_values=AMBIGUITY_TYPES
        )
        
        # Find overrepresented ambiguity types
        overrep_types = self.get_overrepresented("ambiguity_types", threshold=2.0)
        
        # Find underrepresented templates
        underrep_templates = self.get_underrepresented(
            "ambiguity_templates", threshold=0.5, all_values=AMBIGUITY_TEMPLATES
        )
        
        # Find overrepresented templates
        overrep_templates = self.get_overrepresented("ambiguity_templates", threshold=2.0)
        
        # Determine strength based on samples
        if self.total_samples < 10:
            strength = 0.3
        elif self.total_samples < 50:
            strength = 0.5
        else:
            strength = 0.7
        
        return {
            "preferred_types": underrep_types[:2] if underrep_types else [],
            "avoid_types": overrep_types[:2] if overrep_types else [],
            "preferred_templates": underrep_templates[:4] if underrep_templates else [],
            "avoid_templates": overrep_templates[:2] if overrep_templates else [],
            "strength": strength
        }
    
    def get_diversity_report(self, all_chart_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a diversity report.
        
        Args:
            all_chart_types: Optional list of all possible chart types (for complete distribution).
                           If provided, the feature_space will include all chart types with 0 counts for unused ones.
        """
        # Build complete feature_space with all possible chart types
        feature_space_dict = self.feature_space.to_dict()
        
        # If all_chart_types is provided, ensure all types are included (even with 0 count)
        if all_chart_types:
            complete_chart_types = {ct: 0 for ct in all_chart_types}
            # Update with actual counts
            complete_chart_types.update(feature_space_dict.get("chart_types", {}))
            feature_space_dict["chart_types"] = complete_chart_types
        
        return {
            "total_samples": self.total_samples,
            "overall_diversity_score": self.get_overall_diversity_score(),
            "dimension_scores": {
                "encoding_patterns": self.calculate_diversity_score("encoding_patterns"),
                "aggregations": self.calculate_diversity_score("aggregations"),
                "time_units": self.calculate_diversity_score("time_units"),
                "transforms": self.calculate_diversity_score("transforms"),
                "compositions": self.calculate_diversity_score("compositions"),
                "interactions": self.calculate_diversity_score("interactions"),
                "domains": self.calculate_diversity_score("domains"),
                "chart_types": self.calculate_diversity_score("chart_types"),
                "chart_categories": self.calculate_diversity_score("chart_categories"),
            },
            "feature_space": feature_space_dict,
            "underrepresented": {
                "aggregations": self.get_underrepresented("aggregations", all_values=AGGREGATION_TYPES),
                "time_units": self.get_underrepresented("time_units", all_values=TIME_UNITS),
                "transforms": self.get_underrepresented("transforms", all_values=TRANSFORM_TYPES),
                "compositions": self.get_underrepresented("compositions", all_values=COMPOSITION_TYPES),
            },
            "chart_type_stats": {
                "total_chart_types_available": len(all_chart_types) if all_chart_types else len(self.feature_space.chart_types),
                "total_chart_types_used": len(self.feature_space.chart_types),
                "total_categories_used": len(self.feature_space.chart_categories),
                "top_chart_types": dict(sorted(
                    self.feature_space.chart_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                "overrepresented": self.get_overrepresented_chart_types(),
            },
            "ambiguity_stats": {
                "types_distribution": dict(self.feature_space.ambiguity_types),
                "templates_distribution": dict(self.feature_space.ambiguity_templates),
                "underrepresented_types": self.get_underrepresented("ambiguity_types", all_values=AMBIGUITY_TYPES),
                "underrepresented_templates": self.get_underrepresented("ambiguity_templates", all_values=AMBIGUITY_TEMPLATES),
            }
        }
    
    def infer_domain(self, db_id: str) -> str:
        """Infer data domain from database ID."""
        db_lower = db_id.lower()
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in db_lower for kw in keywords):
                return domain
        
        return "other"
    
    def rebuild_from_samples(self, samples: List[Union[Dict[str, Any], 'VisSample']]):
        """
        Rebuild diversity tracker state from a list of samples.
        Used for restoring state from samples_full.json after interruption.
        
        Args:
            samples: List of samples (can be dict from JSON or VisSample instances)
        """
        # Clear current state
        self.feature_space = FeatureSpace()
        self.total_samples = 0
        self._sample_ids.clear()
        
        # Record each sample
        for sample in samples:
            try:
                if isinstance(sample, dict):
                    # Convert dict to a minimal object for tracking
                    # We only need the fields that record_sample() uses
                    sample_obj = self._dict_to_sample_for_tracking(sample)
                else:
                    # Already a VisSample instance
                    sample_obj = sample
                
                self.record_sample(sample_obj)
            except Exception as e:
                logger.warning(f"Failed to rebuild from sample {sample.get('sample_id', 'unknown') if isinstance(sample, dict) else getattr(sample, 'sample_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Rebuilt diversity tracker from {len(samples)} samples")
    
    def _dict_to_sample_for_tracking(self, sample_dict: Dict[str, Any]) -> 'VisSample':
        """
        Convert a dict (from JSON) to a minimal VisSample-like object for tracking.
        Only extracts fields needed by record_sample().
        """
        from core.models import VisSample, Feature, AmbiguityProfile
        
        # Extract key features (support both 'features' and 'required_key_features')
        features_data = sample_dict.get("features", [])
        required_features_data = sample_dict.get("required_key_features", sample_dict.get("key_features", []))
        
        # Convert to Feature objects
        features = [Feature.from_dict(f) for f in features_data]
        required_features = []
        for f in required_features_data:
            # RequiredKeyFeature has a different structure, convert if needed
            if isinstance(f, dict):
                # Try to create a Feature from it (RequiredKeyFeature extends Feature)
                try:
                    from core.models import RequiredKeyFeature
                    required_features.append(RequiredKeyFeature.from_dict(f))
                except:
                    # Fallback to regular Feature
                    required_features.append(Feature.from_dict(f))
        
        # Extract ambiguity profile
        ambiguity_profile = None
        if sample_dict.get("ambiguity_profile"):
            ambiguity_profile = AmbiguityProfile.from_dict(sample_dict["ambiguity_profile"])
        
        # Create a minimal VisSample instance
        # We only need the fields that record_sample() accesses
        sample = VisSample(
            sample_id=sample_dict.get("sample_id", ""),
            source_question_id=sample_dict.get("source_question_id", 0),
            candidate_id=sample_dict.get("candidate_id", ""),
            vis_sql=sample_dict.get("vis_sql", ""),
            vis_question=sample_dict.get("vis_question", ""),
            vis_question_clear=sample_dict.get("vis_question_clear", ""),
            original_question=sample_dict.get("original_question", ""),
            ambiguity_profile=ambiguity_profile,
            data=sample_dict.get("data", []),
            data_processing_code=sample_dict.get("data_processing_code", ""),
            altair_code=sample_dict.get("altair_code", ""),
            vega_lite_spec=sample_dict.get("vega_lite_spec", {}),
            image_path=sample_dict.get("image_path"),
            features=features,
            required_key_features=required_features,
            chart_type=sample_dict.get("chart_type", ""),
            chart_category=sample_dict.get("chart_category", ""),
            vis_intent_type=sample_dict.get("vis_intent_type", ""),
            db_id=sample_dict.get("db_id", ""),
            difficulty=sample_dict.get("difficulty", ""),
        )
        
        return sample
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export diversity tracker state to dictionary.
        Used for saving to dataset_statistic.json.
        
        Returns:
            Dictionary with complete diversity state
        """
        return {
            "total_samples": self.total_samples,
            "sample_ids": sorted(list(self._sample_ids)),  # Sort for consistency
            "feature_space": self.feature_space.to_dict(),
            "overall_diversity_score": self.get_overall_diversity_score(),
            "dimension_scores": {
                "encoding_patterns": self.calculate_diversity_score("encoding_patterns"),
                "aggregations": self.calculate_diversity_score("aggregations"),
                "time_units": self.calculate_diversity_score("time_units"),
                "transforms": self.calculate_diversity_score("transforms"),
                "compositions": self.calculate_diversity_score("compositions"),
                "interactions": self.calculate_diversity_score("interactions"),
                "domains": self.calculate_diversity_score("domains"),
                "chart_types": self.calculate_diversity_score("chart_types"),
                "chart_categories": self.calculate_diversity_score("chart_categories"),
            }
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """
        Restore diversity tracker state from dictionary.
        Used for loading from dataset_statistic.json.
        
        Args:
            data: Dictionary with diversity state
        """
        self.total_samples = data.get("total_samples", 0)
        self._sample_ids = set(data.get("sample_ids", []))
        
        # Restore feature_space
        feature_space_data = data.get("feature_space", {})
        if feature_space_data:
            self.feature_space = FeatureSpace()
            for key, value in feature_space_data.items():
                if hasattr(self.feature_space, key):
                    attr = getattr(self.feature_space, key)
                    if isinstance(attr, dict):
                        attr.clear()
                        attr.update(value)
        
        logger.info(f"Restored diversity tracker state: {self.total_samples} samples")


# Singleton instance
_tracker: Optional[DiversityTracker] = None


def get_diversity_tracker() -> DiversityTracker:
    """Get or create singleton diversity tracker."""
    global _tracker
    if _tracker is None:
        _tracker = DiversityTracker()
    return _tracker


def reset_diversity_tracker():
    """Reset the diversity tracker (for testing)."""
    global _tracker
    _tracker = DiversityTracker()
