"""
Preprocessor for Vis-Interact Dataset Construction.

Handles BIRD Mini-Dev data loading, schema building, and semantic analysis.
This is Step 0 of the pipeline - preparing all offline assets.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from core.models import (
    DataInstance, DatabaseInfo, SQLSemanticSummary,
    SemanticContext, VisPotential, PreprocessedAssets
)
from core.config import get_config
from .schema import get_schema_processor
from .sql_analyzer import get_sql_processor

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Step 0: Offline asset preprocessing for BIRD Mini-Dev.
    
    Responsibilities:
    - Load DataInstances from mini_dev_sqlite.json
    - Build schema catalogs from dev_tables.json
    - Perform semantic analysis on SQL and NL
    - Analyze visualization potential
    """
    
    def __init__(self):
        self.config = get_config()
        self.schema_processor = get_schema_processor()
        self.sql_processor = get_sql_processor()
        
        self._instances: List[DataInstance] = []
        self._loaded = False
    
    # =========================================================================
    # Data Instance Loading
    # =========================================================================
    
    def load_data_instances(self) -> List[DataInstance]:
        """Load all DataInstances from BIRD Mini-Dev JSON."""
        if self._loaded:
            return self._instances
        
        json_path = self.config.paths.minidev_json
        
        if not json_path.exists():
            logger.error(f"mini_dev_sqlite.json not found at {json_path}")
            raise FileNotFoundError(f"mini_dev_sqlite.json not found at {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instances = []
        for item in data:
            try:
                instance = DataInstance.from_dict(item)
                instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse instance: {e}")
        
        self._instances = instances
        self._loaded = True
        logger.info(f"Loaded {len(instances)} BIRD Mini-Dev instances")
        
        return instances
    
    def get_instance_by_id(self, question_id: int) -> Optional[DataInstance]:
        """Get a specific instance by question_id."""
        instances = self.load_data_instances()
        for inst in instances:
            if inst.question_id == question_id:
                return inst
        return None
    
    def get_instances_by_db(self, db_id: str) -> List[DataInstance]:
        """Get all instances for a specific database."""
        instances = self.load_data_instances()
        return [inst for inst in instances if inst.db_id == db_id]
    
    def get_instances_by_difficulty(self, difficulty: str) -> List[DataInstance]:
        """Get all instances with a specific difficulty level."""
        instances = self.load_data_instances()
        return [inst for inst in instances if inst.difficulty.lower() == difficulty.lower()]
    
    # =========================================================================
    # Schema Building
    # =========================================================================
    
    def build_schema_catalog(
        self, 
        db_id: str, 
        include_sample_values: bool = False
    ) -> Optional[DatabaseInfo]:
        """Build complete schema catalog for a database."""
        return self.schema_processor.build_schema_catalog(db_id, include_sample_values)
    
    def build_schema_slice(
        self,
        db_info: DatabaseInfo,
        relevant_tables: List[str]
    ) -> DatabaseInfo:
        """Create a schema slice containing only relevant tables."""
        return self.schema_processor.build_schema_slice(db_info, relevant_tables)
    
    # =========================================================================
    # Semantic Analysis
    # =========================================================================
    
    def analyze_instance(self, instance: DataInstance) -> SemanticContext:
        """
        Perform complete semantic analysis on a DataInstance.
        
        This builds the SemanticContext that will be used for:
        - Key Feature Candidate generation
        - SQL transformation
        - NL transformation
        
        Note: We provide the FULL database schema (not sliced) to allow
        LLM to explore all tables and discover diverse visualization candidates.
        Tables are organized by FK relationships using table groups.
        """
        # Build SQL semantic summary
        sql_summary = self.sql_processor.build_semantic_summary(instance.gold_sql)
        
        # Build FULL schema info (no slicing - provide all tables for LLM exploration)
        schema_info = self.build_schema_catalog(instance.db_id)
        
        # NOTE: We no longer slice the schema to allow LLM to discover
        # cross-table analysis opportunities beyond the original SQL
        
        # Extract key entities
        key_entities = self._extract_key_entities(instance.question, instance.evidence)
        
        # Parse domain knowledge from evidence
        domain_knowledge = self._parse_evidence(instance.evidence)
        
        # Analyze visualization potential (kept for backward compatibility, but not used in prompts)
        vis_potential = self._analyze_vis_potential(sql_summary, schema_info)
        
        # Add evidence to schema_info
        if schema_info:
            schema_info.evidence = instance.evidence
        
        return SemanticContext(
            instance=instance,
            sql_summary=sql_summary,
            key_entities=key_entities,
            domain_knowledge=domain_knowledge,
            schema_info=schema_info,
            vis_potential=vis_potential
        )
    
    def _extract_key_entities(self, question: str, evidence: str) -> List[str]:
        """Extract key business entities from question and evidence."""
        entities = []
        
        # Extract quoted strings
        import re
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        entities.extend(quoted)
        
        # Extract capitalized words (potential entity names)
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                # Skip common starting words
                if word.lower() not in ["what", "which", "where", "when", "who", "how", "the", "show"]:
                    entities.append(word)
        
        return list(set(entities))
    
    def _parse_evidence(self, evidence: str) -> List[str]:
        """Parse domain knowledge from evidence string."""
        if not evidence:
            return []
        
        # Split by common separators
        knowledge_points = []
        
        # Split by newlines and semicolons
        parts = evidence.replace("\n", ";").split(";")
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:  # Skip very short fragments
                knowledge_points.append(part)
        
        return knowledge_points
    
    def _analyze_vis_potential(
        self,
        sql_summary: SQLSemanticSummary,
        schema_info: Optional[DatabaseInfo]
    ) -> VisPotential:
        """
        Analyze visualization potential based on SQL and schema.
        
        Identifies:
        - Temporal fields (for time series)
        - Categorical fields (for grouping)
        - Numeric fields (for measures)
        - Potential chart types
        """
        temporal_fields = []
        categorical_fields = []
        numeric_fields = []
        potential_dimensions = []
        potential_measures = []
        
        # Analyze schema columns
        if schema_info:
            for table in schema_info.tables:
                for col in table.columns:
                    semantic_type = self.schema_processor.infer_column_semantics(col)
                    
                    if semantic_type == "temporal":
                        temporal_fields.append(col.name)
                        potential_dimensions.append(col.name)
                    elif semantic_type == "categorical":
                        categorical_fields.append(col.name)
                        potential_dimensions.append(col.name)
                    elif semantic_type == "identifier":
                        # Identifiers are often useful as:
                        # - dimensions (e.g., grouping by an ID-like code)
                        # - COUNT/COUNT(DISTINCT) measures
                        categorical_fields.append(col.name)
                        potential_dimensions.append(col.name)
                        potential_measures.append(col.name)
                    elif semantic_type == "numeric":
                        numeric_fields.append(col.name)
                        potential_measures.append(col.name)
        
        # Enhance from SQL summary
        if sql_summary:
            # GROUP BY fields are dimensions
            for dim in sql_summary.groupby_dims:
                if dim not in potential_dimensions:
                    potential_dimensions.append(dim)
                    categorical_fields.append(dim)
            
            # Aggregated fields are measures
            for measure in sql_summary.measures:
                for arg in measure.get("args", []):
                    if arg not in potential_measures:
                        potential_measures.append(arg)
        
        # Determine potential chart types
        potential_chart_types = self._suggest_chart_types(
            temporal_fields, categorical_fields, numeric_fields, sql_summary
        )
        
        return VisPotential(
            temporal_fields=temporal_fields,
            categorical_fields=categorical_fields,
            numeric_fields=numeric_fields,
            potential_chart_types=potential_chart_types,
            potential_dimensions=potential_dimensions,
            potential_measures=potential_measures
        )
    
    def _suggest_chart_types(
        self,
        temporal_fields: List[str],
        categorical_fields: List[str],
        numeric_fields: List[str],
        sql_summary: Optional[SQLSemanticSummary]
    ) -> List[str]:
        """Suggest potential chart types based on data characteristics."""
        suggestions = []
        
        has_temporal = len(temporal_fields) > 0
        has_categorical = len(categorical_fields) > 0
        has_numeric = len(numeric_fields) > 0
        
        # Check if aggregation is present
        has_aggregation = sql_summary and len(sql_summary.measures) > 0
        has_groupby = sql_summary and len(sql_summary.groupby_dims) > 0
        
        # Time series data
        if has_temporal and has_numeric:
            suggestions.extend(["line_chart", "area_chart"])
        
        # Categorical comparison
        if has_categorical and has_numeric:
            suggestions.extend(["bar_chart", "horizontal_bar_chart"])
        
        # Distribution
        if has_numeric:
            suggestions.append("histogram")
        
        # Multiple numeric variables
        if len(numeric_fields) >= 2:
            suggestions.append("scatter_plot")
        
        # Aggregated data with groups
        if has_aggregation and has_groupby:
            suggestions.append("grouped_bar_chart")
            if len(sql_summary.groupby_dims) > 1:
                suggestions.append("stacked_bar_chart")
        
        # Proportions
        if has_categorical and has_aggregation:
            suggestions.append("pie_chart")
        
        # Default fallbacks
        if not suggestions:
            suggestions = ["bar_chart", "line_chart", "scatter_plot"]
        
        return list(set(suggestions))
    
    # =========================================================================
    # Preprocessed Assets
    # =========================================================================
    
    def prepare_assets(self, instance: DataInstance) -> Optional[PreprocessedAssets]:
        """
        Prepare all preprocessed assets for an instance.
        
        Returns PreprocessedAssets or None if preparation fails.
        """
        try:
            # Build schema
            schema_info = self.build_schema_catalog(instance.db_id)
            if not schema_info:
                logger.warning(f"Failed to build schema for {instance.db_id}")
                return None
            
            # Parse SQL
            sql_summary = self.sql_processor.build_semantic_summary(instance.gold_sql)
            
            # Build schema slice if we have SQL tables
            if sql_summary and sql_summary.tables:
                schema_slice = self.build_schema_slice(schema_info, sql_summary.tables)
            else:
                schema_slice = schema_info
            
            # Analyze vis potential
            vis_potential = self._analyze_vis_potential(sql_summary, schema_slice)
            
            # Add evidence
            schema_slice.evidence = instance.evidence
            
            return PreprocessedAssets(
                schema_info=schema_slice,
                sql_semantic_summary=sql_summary,
                gold_sql=instance.gold_sql,
                evidence=instance.evidence,
                vis_potential=vis_potential
            )
            
        except Exception as e:
            logger.error(f"Asset preparation failed for question_id={instance.question_id}: {e}")
            return None
    
    def prepare_batch_assets(
        self,
        instances: List[DataInstance],
        show_progress: bool = False
    ) -> Dict[int, PreprocessedAssets]:
        """
        Prepare assets for a batch of instances.
        
        Returns dict mapping question_id to PreprocessedAssets.
        """
        results = {}
        
        inst_iter = instances
        if show_progress and tqdm is not None:
            inst_iter = tqdm(instances, desc="Preparing assets", unit="inst")
        
        for instance in inst_iter:
            assets = self.prepare_assets(instance)
            if assets:
                results[instance.question_id] = assets
        
        logger.info(f"Prepared assets for {len(results)}/{len(instances)} instances")
        return results
    
    # =========================================================================
    # Batch Semantic Analysis
    # =========================================================================
    
    def analyze_batch(
        self,
        instances: List[DataInstance],
        show_progress: bool = False
    ) -> Dict[int, SemanticContext]:
        """
        Analyze a batch of instances.
        
        Returns dict mapping question_id to SemanticContext.
        """
        results = {}
        
        inst_iter = instances
        if show_progress and tqdm is not None:
            inst_iter = tqdm(instances, desc="Analyzing instances", unit="inst")
        
        for instance in inst_iter:
            try:
                ctx = self.analyze_instance(instance)
                results[instance.question_id] = ctx
            except Exception as e:
                logger.warning(f"Analysis failed for question_id={instance.question_id}: {e}")
        
        return results
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        instances = self.load_data_instances()
        
        # Count by difficulty
        difficulty_counts = {}
        for inst in instances:
            diff = inst.difficulty or "unknown"
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        # Count by database
        db_counts = {}
        for inst in instances:
            db_counts[inst.db_id] = db_counts.get(inst.db_id, 0) + 1
        
        # Instances with evidence
        with_evidence = sum(1 for inst in instances if inst.evidence)
        
        return {
            "total_instances": len(instances),
            "unique_databases": len(db_counts),
            "difficulty_distribution": difficulty_counts,
            "database_distribution": db_counts,
            "instances_with_evidence": with_evidence,
            "evidence_coverage": f"{with_evidence / len(instances) * 100:.1f}%"
        }


# Singleton instance
_preprocessor: Optional[Preprocessor] = None


def get_preprocessor() -> Preprocessor:
    """Get or create singleton preprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = Preprocessor()
    return _preprocessor
