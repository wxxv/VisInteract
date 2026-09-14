"""
Core data models for Vis-Interact Dataset Construction.
Designed for BIRD Mini-Dev SQLite dataset with one-to-many sample generation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import json


class AmbiguityType(str, Enum):
    """Types of ambiguity injection."""
    DATA = "data_ambiguity"
    VISUALIZATION = "visualization_ambiguity"
    INFO_COMPLETION = "information_completion"
    ERROR_CORRECTION = "error_correction"


class PreferredInterface(str, Enum):
    """Preferred ask_user interface type."""
    TEXT = "ask_user_text"
    VIS = "ask_user_vis"


class Difficulty(str, Enum):
    """Difficulty level for ambiguity resolution."""
    EASY = "easy"      # 0 rounds to clarify
    MEDIUM = "medium"  # 1 round to clarify
    HARD = "hard"      # 2 rounds to clarify


class VisIntentType(str, Enum):
    """Types of visualization intent."""
    TREND = "trend"              # Time series, evolution
    COMPARISON = "comparison"    # Compare categories
    DISTRIBUTION = "distribution"  # Histogram, density
    RANKING = "ranking"          # Top-K, leaderboard
    CORRELATION = "correlation"  # Scatter, relationship
    COMPOSITION = "composition"  # Part-of-whole, pie


class FeatureOp(str, Enum):
    """Operators for feature constraints (used in spec validation)."""
    EXISTS = "exists"
    EQ = "eq"
    IN = "in"
    CONTAINS = "contains"
    LEN_GE = "len_ge"
    LEN_LE = "len_le"


@dataclass
class Feature:
    """
    A single feature constraint in path-op-value format.
    Used for Altair/Vega-Lite spec validation.

    Note: This was previously called KeyFeature. The new RequiredKeyFeature
    class is now used for NL-based evaluation metrics.
    """
    path: str                    # e.g., "encoding.x.field", "mark.type"
    op: FeatureOp                # exists/eq/in/contains/len_ge/len_le
    value: Any = None            # The expected value (None for 'exists' op)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op.value if isinstance(self.op, FeatureOp) else self.op,
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        return cls(
            path=data["path"],
            op=FeatureOp(data["op"]) if isinstance(data["op"], str) else data["op"],
            value=data.get("value")
        )


class RequiredKeyFeatureType(str, Enum):
    """Types of required key features (for non-technical user requirements)."""
    MARK = "mark"                      # Chart type (bar/line/point/area/arc)
    ENCODING = "encoding"              # Required fields/channels (x/y/color/size)
    AGGREGATION = "aggregation"        # Aggregation semantics (sum/mean/count/top-k)
    FILTER = "filter"                  # Data scope/filtering
    OVERLAY_STAT_LINE = "overlay_stat_line"  # Statistical reference lines (mean/median/target)
    INTERACTION = "interaction"        # Interactive features (brush/selection/slider)
    COMPOSITION = "composition"        # Multi-view requirements (layer/facet/concat)


@dataclass
class RequiredKeyFeature:
    """
    A required key feature representing a non-technical user's "hard requirement".
    Uses natural language description as the primary representation.

    Used for Metric evaluation (Code LLM Judge / Chart VLM Judge).
    See Metric.md for full specification.

    Example:
        RequiredKeyFeature(
            id="kf_mark_bar",
            type=RequiredKeyFeatureType.MARK,
            text="Use a bar chart.",
            args={"mark": "bar"},
            must=True
        )
    """
    id: str                                    # e.g., "kf_mark_bar"
    type: RequiredKeyFeatureType               # Feature category
    text: str                                  # NL description (core): "Use a bar chart."
    args: Dict[str, Any] = field(default_factory=dict)  # Optional structured params: {"mark": "bar"}
    must: bool = True                          # If True, cannot be corrupted by ambiguity injection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, RequiredKeyFeatureType) else self.type,
            "text": self.text,
            "args": self.args,
            "must": self.must
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequiredKeyFeature":
        type_val = data.get("type", "mark")
        if isinstance(type_val, str):
            try:
                type_val = RequiredKeyFeatureType(type_val)
            except ValueError:
                type_val = RequiredKeyFeatureType.MARK
        return cls(
            id=data.get("id", ""),
            type=type_val,
            text=data.get("text", ""),
            args=data.get("args", {}),
            must=data.get("must", True)
        )


@dataclass
class AmbiguityProfile:
    """Configuration for ambiguity injection in a sample."""
    ambiguity_types: List[AmbiguityType]  # List of ambiguity types (supports multiple)
    template_names: List[str]  # List of selected template names
    target_feature_paths: List[str]  # Paths to features being made ambiguous
    preferred_interface: PreferredInterface
    difficulty: Difficulty
    corruption: Optional[str] = None  # For error correction type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ambiguity_types": [t.value for t in self.ambiguity_types],
            "template_names": self.template_names,
            "target_feature_paths": self.target_feature_paths,
            "preferred_interface": self.preferred_interface.value,
            "difficulty": self.difficulty.value,
            "corruption": self.corruption,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmbiguityProfile":
        """
        Create AmbiguityProfile from dictionary.
        Only supports the new list-based format (ambiguity_types, template_names).
        """
        return cls(
            ambiguity_types=[AmbiguityType(t) for t in data["ambiguity_types"]],
            template_names=data["template_names"],
            target_feature_paths=data["target_feature_paths"],
            preferred_interface=PreferredInterface(data["preferred_interface"]),
            difficulty=Difficulty(data["difficulty"]),
            corruption=data.get("corruption"),
        )


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    table_name: str = ""
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)
    is_primary_key: bool = False
    is_foreign_key: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "table_name": self.table_name,
            "description": self.description,
            "sample_values": self.sample_values,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key
        }


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "row_count": self.row_count
        }


@dataclass
class ForeignKeyInfo:
    """Foreign key relationship information."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column
        }


@dataclass
class TableGroup:
    """
    A group of tables connected by foreign key relationships.
    Used to organize schema presentation for LLM context.
    """
    tables: List["TableInfo"] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": [t.to_dict() for t in self.tables],
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys]
        }

    @property
    def table_names(self) -> List[str]:
        """Get list of table names in this group."""
        return [t.name for t in self.tables]


@dataclass
class DatabaseInfo:
    """Information about a database for a sample."""
    db_id: str
    tables: List[TableInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    evidence: Optional[str] = None  # Domain knowledge/evidence from BIRD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_id": self.db_id,
            "tables": [t.to_dict() for t in self.tables],
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "evidence": self.evidence
        }

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Get table by name."""
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        return None

    def get_all_columns(self) -> List[ColumnInfo]:
        """Get all columns from all tables."""
        columns = []
        for t in self.tables:
            columns.extend(t.columns)
        return columns


@dataclass
class SQLSemanticSummary:
    """Structured semantic summary of a SQL query."""
    tables: List[str] = field(default_factory=list)
    joins: List[Dict[str, str]] = field(default_factory=list)  # [{left, right, on}]
    filters: List[str] = field(default_factory=list)  # WHERE/HAVING conditions
    groupby_dims: List[str] = field(default_factory=list)
    measures: List[Dict[str, str]] = field(default_factory=list)  # [{func, field}]
    order_limit: Optional[Dict[str, Any]] = None
    window_ops: List[str] = field(default_factory=list)
    ctes: List[str] = field(default_factory=list)
    select_fields: List[str] = field(default_factory=list)
    sql: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": self.tables,
            "joins": self.joins,
            "filters": self.filters,
            "groupby_dims": self.groupby_dims,
            "measures": self.measures,
            "order_limit": self.order_limit,
            "window_ops": self.window_ops,
            "ctes": self.ctes,
            "select_fields": self.select_fields,
            "sql": self.sql
        }


# ============== BIRD Mini-Dev Data Models ==============


@dataclass
class DataInstance:
    """
    A BIRD Mini-Dev SQLite instance.
    Replaces the legacy SpiderInstance.
    """
    question_id: int
    question: str          # Original NL question
    db_id: str
    evidence: str = ""     # Domain knowledge/hints
    gold_sql: str = ""     # Gold SQL from BIRD
    difficulty: str = ""   # simple/moderate/challenging

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataInstance":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            db_id=data["db_id"],
            evidence=data.get("evidence", ""),
            gold_sql=data.get("SQL", ""),
            difficulty=data.get("difficulty", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "db_id": self.db_id,
            "evidence": self.evidence,
            "gold_sql": self.gold_sql,
            "difficulty": self.difficulty
        }


@dataclass
class VisPotential:
    """
    Analysis of visualization potential for a database/SQL context.
    Identifies what kinds of visualizations are possible.
    """
    temporal_fields: List[str] = field(default_factory=list)      # Date/time fields
    categorical_fields: List[str] = field(default_factory=list)   # Categorical fields
    numeric_fields: List[str] = field(default_factory=list)       # Numeric/measure fields
    potential_chart_types: List[str] = field(default_factory=list)
    potential_dimensions: List[str] = field(default_factory=list)
    potential_measures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_fields": self.temporal_fields,
            "categorical_fields": self.categorical_fields,
            "numeric_fields": self.numeric_fields,
            "potential_chart_types": self.potential_chart_types,
            "potential_dimensions": self.potential_dimensions,
            "potential_measures": self.potential_measures
        }


@dataclass
class SemanticContext:
    """
    Complete semantic analysis result for a DataInstance.
    Contains all information needed for Key Feature generation.
    """
    instance: DataInstance
    sql_summary: SQLSemanticSummary
    key_entities: List[str] = field(default_factory=list)
    domain_knowledge: List[str] = field(default_factory=list)
    schema_info: Optional[DatabaseInfo] = None
    vis_potential: Optional[VisPotential] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance.to_dict(),
            "sql_summary": self.sql_summary.to_dict(),
            "key_entities": self.key_entities,
            "domain_knowledge": self.domain_knowledge,
            "schema_info": self.schema_info.to_dict() if self.schema_info else None,
            "vis_potential": self.vis_potential.to_dict() if self.vis_potential else None
        }


@dataclass
class FeatureCandidate:
    """
    A visualization candidate with semantic descriptions.
    Multiple candidates are generated per DataInstance for diversity.

    Note: Previously called KeyFeatureCandidate.

    Design Philosophy:
    - Describes "what" (intent) and "structure" (mapping), not "how" (implementation)
    - Transform and Spec generation stages handle implementation details
    - Supports complex charts through flexible data_mapping structure
    """
    candidate_id: str
    chart_type: str
    chart_category: str = ""  # e.g., "Bar Charts", "Interactive Charts"

    # ===== Intent & Context =====
    vis_intent: str = ""
    vis_intent_type: VisIntentType = VisIntentType.COMPARISON
    analytical_question: str = ""
    tables_needed: List[str] = field(default_factory=list)

    # ===== Data Mapping (flexible structure for complex charts) =====
    data_mapping: Dict[str, Any] = field(default_factory=dict)
    # Example:
    # {
    #   "primary": {"x": "Date (temporal)", "y": "Revenue (sum)", "color": "Region"},
    #   "secondary": {"y": "Profit (sum)"},  # for dual-axis
    #   "layers": [{"type": "reference_line", "y": "Target"}],
    #   "facet": {"field": "Region", "type": "row"}
    # }

    # ===== Visualization Features (semantic descriptions) =====
    visualization_features: List[str] = field(default_factory=list)
    # Example:
    # [
    #   "dual_y_axis: independent scales for revenue and profit",
    #   "reference_line: show quarterly targets",
    #   "facet: split by region",
    #   "interaction: hover tooltip with all metrics"
    # ]

    # ===== Basic fields (kept for simple scenarios and backward compatibility) =====
    dimensions: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)
    aggregation: str = "COUNT"        # SUM/AVG/COUNT/etc

    # ===== Features (for spec validation) =====
    features: List[Feature] = field(default_factory=list)

    # Diversity metrics
    diversity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chart_type": self.chart_type,
            "chart_category": self.chart_category,
            "vis_intent": self.vis_intent,
            "vis_intent_type": self.vis_intent_type.value,
            "analytical_question": self.analytical_question,
            "tables_needed": self.tables_needed,
            "data_mapping": self.data_mapping,
            "visualization_features": self.visualization_features,
            "dimensions": self.dimensions,
            "measures": self.measures,
            "aggregation": self.aggregation,
            "features": [f.to_dict() for f in self.features],
            "diversity_score": self.diversity_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureCandidate":
        # Support both "features" and legacy "key_features" field names
        features_data = data.get("features", data.get("key_features", []))
        return cls(
            candidate_id=data.get("candidate_id", ""),
            chart_type=data.get("chart_type", "bar_chart"),
            chart_category=data.get("chart_category", ""),
            vis_intent=data.get("vis_intent", ""),
            vis_intent_type=VisIntentType(data.get("vis_intent_type", "comparison")),
            analytical_question=data.get("analytical_question", ""),
            tables_needed=data.get("tables_needed", []),
            data_mapping=data.get("data_mapping", {}),
            visualization_features=data.get("visualization_features", []),
            dimensions=data.get("dimensions", []),
            measures=data.get("measures", []),
            aggregation=data.get("aggregation", "COUNT"),
            features=[Feature.from_dict(f) for f in features_data],
            diversity_score=data.get("diversity_score", 0.0)
        )

    @property
    def has_advanced_effects(self) -> bool:
        """Check if this candidate uses advanced visualization effects."""
        return bool(
            self.visualization_features or
            (self.data_mapping and len(self.data_mapping) > 1) or
            "secondary" in self.data_mapping or
            "layers" in self.data_mapping or
            "facet" in self.data_mapping
        )


@dataclass
class VisSample:
    """
    A complete visualization sample.
    One DataInstance can produce multiple VisSamples (one per candidate).
    """
    # Identifiers
    sample_id: str
    source_question_id: int
    candidate_id: str

    # Source reference
    source_instance: Optional[DataInstance] = None
    candidate: Optional[FeatureCandidate] = None

    # Transformed content
    vis_sql: str = ""  # Deprecated: use data_processing_code instead
    vis_question: str = ""
    vis_question_clear: str = ""
    original_question: str = ""

    # Ambiguity
    ambiguity_profile: Optional[AmbiguityProfile] = None

    # Data
    data: List[Dict[str, Any]] = field(default_factory=list)

    # Generated artifacts
    data_processing_code: str = ""  # Complete data processing code (import + SQL + Pandas)
    altair_code: str = ""  # Complete code (data processing + Altair visualization)
    vega_lite_spec: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None

    # Features (for spec validation, path-op-value format)
    features: List[Feature] = field(default_factory=list)

    # Required Key Features (NL-based, for metric evaluation)
    required_key_features: List[RequiredKeyFeature] = field(default_factory=list)

    # Metadata
    chart_type: str = ""
    chart_category: str = ""
    vis_intent_type: str = ""
    db_id: str = ""
    difficulty: str = ""

    # Processing metadata
    transform_log: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = False

    # Validation reports
    data_quality_report: Optional[Dict[str, Any]] = None
    vlm_validation_report: Optional[Dict[str, Any]] = None
    chart_image_path: Optional[str] = None

    def to_dict(self, format: str = "full", compact: bool = None) -> Dict[str, Any]:
        """
        Convert sample to dictionary.
        
        Args:
            format: "full" | "compact" | "export"
                - full: 完整信息（用于samples_full.json和diversity恢复），不包含validation_passed和transform_log
                - compact: 精简格式（用于samples.json发布）
                - export: 最小化发布格式
            compact: Deprecated, use format="compact" instead. Kept for backward compatibility.
        """
        # Backward compatibility: support old compact parameter
        if compact is not None:
            format = "compact" if compact else "full"
        
        if format == "compact":
            # Compact format with only essential fields for dataset release
            return {
                "sample_id": self.sample_id,
                "source_question_id": self.source_question_id,
                "candidate_id": self.candidate_id,
                "db_id": self.db_id,
                "data_processing_code": self.data_processing_code,
                "vis_question": self.vis_question,
                "vis_question_clear": self.vis_question_clear,
                "original_question": self.original_question,
                "altair_code": self.altair_code,
                "ambiguity_profile": self.ambiguity_profile.to_dict() if self.ambiguity_profile else None,
                "key_features": [f.to_dict() for f in self.required_key_features],
                "chart_type": self.chart_type,
                "chart_category": self.chart_category,
                "difficulty": self.difficulty,
            }
        
        if format == "export":
            # Minimal export format
            return self.to_export_dict()
        
        # Full format (default) - for samples_full.json and diversity tracker recovery
        # NOTE: Does NOT include transform_log (debug field) or data (redundant with spec)
        # INCLUDES validation_passed for resume logic to filter out failed samples
        # Truncate datasets in vega_lite_spec to save space
        truncated_spec = self._truncate_spec_datasets(self.vega_lite_spec)
        
        return {
            "sample_id": self.sample_id,
            "source_question_id": self.source_question_id,
            "candidate_id": self.candidate_id,
            "source_instance": self.source_instance.to_dict() if self.source_instance else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "vis_sql": self.vis_sql,
            "data_processing_code": self.data_processing_code,
            "vis_question": self.vis_question,
            "vis_question_clear": self.vis_question_clear,
            "original_question": self.original_question,
            "ambiguity_profile": self.ambiguity_profile.to_dict() if self.ambiguity_profile else None,
            "altair_code": self.altair_code,
            "vega_lite_spec": truncated_spec,
            "image_path": self.image_path,
            "features": [f.to_dict() for f in self.features],
            "required_key_features": [f.to_dict() for f in self.required_key_features],
            "chart_type": self.chart_type,
            "chart_category": self.chart_category,
            "vis_intent_type": self.vis_intent_type,
            "db_id": self.db_id,
            "difficulty": self.difficulty,
            "validation_passed": self.validation_passed,  # Include for resume logic
            "data_quality_report": self.data_quality_report,
            "vlm_validation_report": self.vlm_validation_report,
            "chart_image_path": self.chart_image_path
        }
    
    def _truncate_spec_datasets(self, spec: Dict[str, Any], max_rows: int = 5) -> Dict[str, Any]:
        """
        Truncate datasets in vega_lite_spec to save space.
        Only keeps first max_rows of inline data.
        
        Args:
            spec: Vega-Lite specification
            max_rows: Maximum number of rows to keep in datasets
            
        Returns:
            Spec with truncated datasets
        """
        if not spec:
            return spec
        
        # Create a copy to avoid modifying the original
        import copy
        spec_copy = copy.deepcopy(spec)
        
        # Truncate inline data in spec root
        if "data" in spec_copy and isinstance(spec_copy["data"], dict):
            if "values" in spec_copy["data"] and isinstance(spec_copy["data"]["values"], list):
                original_len = len(spec_copy["data"]["values"])
                if original_len > max_rows:
                    spec_copy["data"]["values"] = spec_copy["data"]["values"][:max_rows]
                    # Add a note about truncation
                    spec_copy["data"]["_truncated"] = True
                    spec_copy["data"]["_original_rows"] = original_len
        
        # Truncate datasets field (used in layered/concat charts)
        if "datasets" in spec_copy and isinstance(spec_copy["datasets"], dict):
            for dataset_name, dataset_values in spec_copy["datasets"].items():
                if isinstance(dataset_values, list):
                    original_len = len(dataset_values)
                    if original_len > max_rows:
                        spec_copy["datasets"][dataset_name] = dataset_values[:max_rows]
                        # Note: Can't add metadata to list, so we create a dict wrapper
                        # But this might break compatibility, so just truncate silently
        
        return spec_copy

    def to_export_dict(self) -> Dict[str, Any]:
        """Export a minimal, display-oriented schema for dataset release."""
        return {
            "sample_id": self.sample_id,
            "source_question_id": self.source_question_id,
            "candidate_id": self.candidate_id,
            "vis_sql": self.vis_sql,
            "data_processing_code": self.data_processing_code,
            "vis_question": self.vis_question,
            "vis_question_clear": self.vis_question_clear,
            "original_question": self.original_question,
            "altair_code": self.altair_code,
            "ambiguity_profile": self.ambiguity_profile.to_dict() if self.ambiguity_profile else None,
            "features": [f.to_dict() for f in self.features],
            "required_key_features": [f.to_dict() for f in self.required_key_features],
            "chart_type": self.chart_type,
            "chart_category": self.chart_category,
            "difficulty": self.difficulty,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert sample to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def key_features(self) -> List[Feature]:
        """Backward compatibility: alias for features."""
        return self.features

    @key_features.setter
    def key_features(self, value: List[Feature]):
        """Backward compatibility: alias for features."""
        self.features = value


# ============== Legacy Models (for backward compatibility) ==============


@dataclass
class ChartContract:
    """Contract defining requirements for a chart type."""
    chart_type: str
    category: str
    must_have: List[Feature] = field(default_factory=list)
    optional: List[Feature] = field(default_factory=list)
    min_data_requirements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "category": self.category,
            "must_have": [f.to_dict() for f in self.must_have],
            "optional": [f.to_dict() for f in self.optional],
            "min_data_requirements": self.min_data_requirements
        }


@dataclass
class InjectionLog:
    """Log of ambiguity injection actions."""
    removed_info: List[str] = field(default_factory=list)
    added_ambiguity: List[str] = field(default_factory=list)
    error_injected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "removed_info": self.removed_info,
            "added_ambiguity": self.added_ambiguity,
            "error_injected": self.error_injected
        }


@dataclass
class PreprocessedAssets:
    """Container for preprocessed assets for an instance."""
    schema_info: Optional[DatabaseInfo] = None
    sql_semantic_summary: Optional[SQLSemanticSummary] = None
    gold_sql: Optional[str] = None
    evidence: Optional[str] = None
    vis_potential: Optional[VisPotential] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_info": self.schema_info.to_dict() if self.schema_info else None,
            "sql_semantic_summary": self.sql_semantic_summary.to_dict() if self.sql_semantic_summary else None,
            "gold_sql": self.gold_sql,
            "evidence": self.evidence,
            "vis_potential": self.vis_potential.to_dict() if self.vis_potential else None
        }


# ============== Iteration Feedback Models ==============


@dataclass
class IterationFeedback:
    """
    Structured feedback from validator for guiding iteration.
    Tells the LLM exactly what to fix and how.
    """
    needs_iteration: bool = False
    iteration_target: str = "none"  # sql/altair_code/both/none

    # SQL related feedback
    sql_issues: List[str] = field(default_factory=list)
    sql_suggestions: List[str] = field(default_factory=list)
    data_issues: List[str] = field(default_factory=list)

    # Altair code related feedback
    code_issues: List[str] = field(default_factory=list)
    code_suggestions: List[str] = field(default_factory=list)
    missing_features: List["Feature"] = field(default_factory=list)
    wrong_features: List[Tuple[str, Any, Any]] = field(default_factory=list)

    # VLM visual feedback
    visual_issues: List[str] = field(default_factory=list)
    visual_suggestions: List[str] = field(default_factory=list)

    # Ambiguity feedback
    ambiguity_issues: List[str] = field(default_factory=list)

    # Priority and control
    priority: str = "medium"  # high/medium/low
    max_retries_reached: bool = False

    @property
    def all_issues(self) -> List[str]:
        return (
            self.sql_issues + self.data_issues +
            self.code_issues + self.visual_issues +
            self.ambiguity_issues
        )

    @property
    def all_suggestions(self) -> List[str]:
        return self.sql_suggestions + self.code_suggestions + self.visual_suggestions

    def to_prompt_context(self) -> str:
        lines = []

        if self.sql_issues:
            lines.append("## SQL Issues")
            for issue in self.sql_issues:
                lines.append(f"- {issue}")
            if self.sql_suggestions:
                lines.append("\n## SQL Fix Suggestions")
                for sug in self.sql_suggestions:
                    lines.append(f"- {sug}")

        if self.data_issues:
            lines.append("\n## Data Issues")
            for issue in self.data_issues:
                lines.append(f"- {issue}")

        if self.code_issues:
            lines.append("\n## Code Issues")
            for issue in self.code_issues:
                lines.append(f"- {issue}")
            if self.code_suggestions:
                lines.append("\n## Code Fix Suggestions")
                for sug in self.code_suggestions:
                    lines.append(f"- {sug}")

        if self.missing_features:
            lines.append("\n## Missing Key Features")
            for f in self.missing_features:
                lines.append(f"- {f.path}: {f.op.value} {f.value if f.value else ''}")

        if self.wrong_features:
            lines.append("\n## Wrong Feature Values")
            for path, expected, actual in self.wrong_features:
                lines.append(f"- {path}: expected {expected}, got {actual}")

        if self.visual_issues:
            lines.append("\n## Visual Issues (from VLM)")
            for issue in self.visual_issues:
                lines.append(f"- {issue}")
            if self.visual_suggestions:
                lines.append("\n## Visual Fix Suggestions")
                for sug in self.visual_suggestions:
                    lines.append(f"- {sug}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "needs_iteration": self.needs_iteration,
            "iteration_target": self.iteration_target,
            "sql_issues": self.sql_issues,
            "sql_suggestions": self.sql_suggestions,
            "data_issues": self.data_issues,
            "code_issues": self.code_issues,
            "code_suggestions": self.code_suggestions,
            "missing_features": [f.to_dict() for f in self.missing_features],
            "wrong_features": self.wrong_features,
            "visual_issues": self.visual_issues,
            "visual_suggestions": self.visual_suggestions,
            "ambiguity_issues": self.ambiguity_issues,
            "priority": self.priority,
            "max_retries_reached": self.max_retries_reached
        }


@dataclass
class IterationContext:
    """Context for an iteration attempt."""
    original_code: str
    feedback: IterationFeedback
    attempt: int
    previous_attempts: List[Tuple[str, "IterationFeedback"]] = field(default_factory=list)

    def get_history_prompt(self) -> str:
        if not self.previous_attempts:
            return ""

        lines = ["## Iteration History"]
        for i, (code, fb) in enumerate(self.previous_attempts, 1):
            lines.append(f"\n### Attempt {i}")
            lines.append(f"```\n{code[:500]}{'...' if len(code) > 500 else ''}\n```")
            if fb.all_issues:
                lines.append("Issues: " + "; ".join(fb.all_issues[:3]))

        lines.append(f"\n## Current Attempt: {self.attempt + 1}")
        lines.append("Please avoid repeating previous errors and try a different approach.")

        return "\n".join(lines)


@dataclass
class IterationLog:
    """Log of iteration attempts during sample generation."""
    sql_attempts: int = 0
    sql_fixes: List[str] = field(default_factory=list)
    code_attempts: int = 0
    code_fixes: List[str] = field(default_factory=list)
    validation_retries: int = 0
    total_iterations: int = 0
    final_status: str = ""
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_iteration(self, iteration: int, target: str, issues: List[str], success: bool):
        self.iteration_history.append({
            "iteration": iteration,
            "target": target,
            "issues": issues,
            "success": success
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql_attempts": self.sql_attempts,
            "sql_fixes": self.sql_fixes,
            "code_attempts": self.code_attempts,
            "code_fixes": self.code_fixes,
            "validation_retries": self.validation_retries,
            "total_iterations": self.total_iterations,
            "final_status": self.final_status,
            "iteration_history": self.iteration_history
        }


@dataclass
class ValidationResult:
    """Result of sample validation."""
    passed: bool
    failure_codes: List[str] = field(default_factory=list)
    data_validation_report: Dict[str, Any] = field(default_factory=dict)
    vis_validation_report: Dict[str, Any] = field(default_factory=dict)
    vlm_validation_report: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    iteration_feedback: Optional[IterationFeedback] = None


