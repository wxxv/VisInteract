"""
Feature Generator for Vis-Interact Dataset Construction.

Generates N diverse Feature Candidates per DataInstance.
Each candidate represents a different visualization approach for the same data.

Note: This module was previously called key_feature_generator.
The class KeyFeatureGenerator is kept for backward compatibility.

Key Design:
1. LLM-driven divergent generation of 8-10 initial candidates
2. Diversity maximization algorithm to select top N (default 5)
3. Each candidate includes: chart_type, dimensions, measures, aggregation, vis_intent
"""
import json
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from core.models import (
    Feature, FeatureOp, FeatureCandidate, VisIntentType,
    SemanticContext, VisPotential, DatabaseInfo, SQLSemanticSummary,
)
from utils.chart_contracts import get_contract_manager
from utils.llm_client import get_llm_client
from preprocessing.schema import get_schema_processor
from core.config import get_config
from utils.diversity import DiversityConstraints
from execution.sqlite_client import get_sqlite_client
import pandas as pd

logger = logging.getLogger(__name__)


class KeyFeatureGenerator:
    """
    Generates diverse Feature Candidates for visualization.
    
    Note: Class name kept as KeyFeatureGenerator for backward compatibility.
    Internally uses FeatureCandidate (formerly KeyFeatureCandidate).
    
    Main API:
    - generate_candidates(): Generate N diverse candidates from SemanticContext
    - maximize_diversity(): Select most diverse subset from candidates
    """
    
    # Visualization intent types
    VIS_INTENT_TYPES = [
        "trend",        # Time series, evolution over time
        "comparison",   # Compare categories/groups
        "distribution", # Histogram, density, spread
        "ranking",      # Top-K, leaderboard, sorted
        "correlation",  # Scatter, relationship between variables
        "composition",  # Part-of-whole, pie, stacked
    ]
    
    # Common aggregation functions
    AGGREGATIONS = ["SUM", "AVG", "COUNT", "MIN", "MAX", "MEDIAN"]

    # =========================================================================
    # Chart Catalog: Dynamically loaded from chart_example directory
    # =========================================================================
    # CHART_CATALOG, CHART_CATEGORIES, and ALL_CHART_TYPES are now loaded
    # dynamically in __init__() using ChartCatalogLoader
    
    def __init__(self):
        self.contract_manager = get_contract_manager()
        self.llm = get_llm_client()
        
        # NEW: Load chart catalog dynamically
        from utils.chart_catalog import get_chart_catalog_loader
        config = get_config()
        self.catalog_loader = get_chart_catalog_loader(config.paths.chart_example_dir)
        self.CHART_CATALOG = self.catalog_loader.load_catalog()
        self.ALL_CHART_TYPES = self.catalog_loader.get_all_chart_types()
        self.CHART_CATEGORIES = self.catalog_loader.get_all_categories()
        
        # Build reverse mapping: chart_type -> category
        self._type_to_category = {}
        for category, types in self.CHART_CATALOG.items():
            for chart_type in types:
                self._type_to_category[chart_type] = category
    
    def get_category_for_type(self, chart_type: str) -> str:
        """Get the category for a chart type."""
        return self._type_to_category.get(chart_type, "Other")
    
    def get_types_for_category(self, category: str) -> List[str]:
        """Get all chart types in a category."""
        return self.CHART_CATALOG.get(category, [])
    
    def _sample_chart_types_for_prompt(
        self,
        num_types: int = 5,
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> List[str]:
        """
        Sample chart types for prompt based on diversity constraints.
        
        This method extracts the sampling logic to be reusable for both
        text formatting and image generation.
        
        Priority:
        1. Diversity constraints (preferred_chart_types)
        2. Random from underrepresented categories
        3. Ensure category diversity
        
        Args:
            num_types: Number of chart types to sample (default 5)
            diversity_constraints: Global diversity constraints from tracker
            
        Returns:
            List of sampled chart type names
        """
        selected_types = []
        
        # Step 1: Start with preferred types from diversity constraints
        if diversity_constraints and diversity_constraints.preferred_chart_types:
            # Sample from preferred types (up to num_types // 2)
            preferred = diversity_constraints.preferred_chart_types
            sample_size = min(len(preferred), num_types // 2)
            selected_types.extend(random.sample(preferred, sample_size))
        
        # Step 2: Add random types from underrepresented categories
        if diversity_constraints and diversity_constraints.preferred_chart_categories:
            remaining = num_types - len(selected_types)
            if remaining > 0:
                for category in diversity_constraints.preferred_chart_categories:
                    if len(selected_types) >= num_types:
                        break
                    types_in_cat = self.catalog_loader.get_types_for_category(category)
                    # Filter out already selected types
                    available = [t for t in types_in_cat if t not in selected_types]
                    if available:
                        selected_types.append(random.choice(available))
        
        # Step 3: Fill up to num_types with random types from ALL catalog
        remaining = num_types - len(selected_types)
        if remaining > 0:
            all_types = self.catalog_loader.get_all_chart_types()
            available = [t for t in all_types if t not in selected_types]
            if available:
                sample_size = min(len(available), remaining)
                selected_types.extend(random.sample(available, sample_size))
        
        # Step 4: Deduplicate similar types to prevent confusion
        selected_types = self._deduplicate_similar_types(selected_types)
        
        return selected_types
    
    def _execute_example_charts(
        self,
        selected_chart_types: List[str],
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute example code for selected chart types and generate images.
        
        Uses ChartExecutor to execute Altair example code and render to PNG images.
        Failed executions are logged but don't stop the process.
        
        Args:
            selected_chart_types: List of chart type names to execute
            diversity_constraints: Global diversity constraints (for metadata)
            
        Returns:
            List of dicts with:
            - chart_type: str (chart type name)
            - category: str (chart category)
            - image_base64: Optional[str] (base64-encoded PNG, None if failed)
            - error: Optional[str] (error message if failed, None if successful)
            - code: str (original example code)
        """
        import os
        import multiprocessing
        from utils.chart_executor import get_chart_executor
        
        config = get_config()
        
        # Check if execution is enabled
        if not config.pipeline.execute_chart_examples:
            logger.debug("Chart example execution is disabled in config")
            return []
        
        # SAFETY: Disable in multiprocessing environment to avoid deadlocks
        # Chart rendering with Altair may not be safe in forked processes
        try:
            current_process = multiprocessing.current_process()
            if current_process.name != 'MainProcess':
                logger.warning(f"Chart execution disabled in worker process '{current_process.name}' to avoid deadlocks")
                return []
        except Exception as e:
            logger.debug(f"Could not check process name: {e}")
        
        # Get executor with configured timeout and scale
        executor = get_chart_executor(
            timeout=config.pipeline.chart_execution_timeout,
            scale=config.pipeline.example_image_scale
        )
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Limit number of executions
        max_executions = min(len(selected_chart_types), config.pipeline.max_example_images)
        
        for chart_type in selected_chart_types[:max_executions]:
            # Get example code
            example_code = self.catalog_loader.get_example_code(chart_type)
            
            if not example_code:
                logger.warning(f"No example code found for chart type: {chart_type}")
                results.append({
                    "chart_type": chart_type,
                    "category": self.get_category_for_type(chart_type),
                    "image_base64": None,
                    "error": "No example code found",
                    "code": ""
                })
                failed_count += 1
                continue
            
            # Execute and generate image
            image_base64, error = executor.execute_chart_example(chart_type, example_code)
            
            results.append({
                "chart_type": chart_type,
                "category": self.get_category_for_type(chart_type),
                "image_base64": image_base64,
                "error": error,
                "code": example_code
            })
            
            if image_base64:
                successful_count += 1
            else:
                failed_count += 1
        
        logger.info(f"Executed {len(results)} example charts: "
                   f"{successful_count} successful, {failed_count} failed")
        
        return results
    
    def sample_and_format_chart_types(
        self,
        num_types: int = 5,
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> str:
        """
        Sample 3-5 chart types with full example code.
        
        Priority:
        1. Diversity constraints (preferred_chart_types)
        2. Random from underrepresented categories
        3. Ensure category diversity
        
        Args:
            num_types: Number of chart types to sample (default 5)
            diversity_constraints: Global diversity constraints from tracker
            
        Returns:
            Formatted string with chart types and example code
        """
        # Use the extracted sampling method
        selected_types = self._sample_chart_types_for_prompt(num_types, diversity_constraints)
        
        # Format each selected type with full example code
        formatted_parts = []
        for chart_type in selected_types:
            formatted = self.catalog_loader.format_example_with_metadata(chart_type)
            if formatted:
                formatted_parts.append(formatted)
        
        if not formatted_parts:
            # Fallback: if no examples found, just list categories
            return "\n".join(f"- **{cat}**" for cat in self.CHART_CATEGORIES[:5])
        
        return "\n\n".join(formatted_parts)
    
    def _deduplicate_similar_types(self, types: List[str]) -> List[str]:
        """
        Remove similar chart types, keeping only one from each similar group.
        
        This prevents the LLM from seeing multiple similar types in the same prompt,
        which could lead to confusion or mixed usage.
        
        Args:
            types: List of chart type names
            
        Returns:
            Deduplicated list with only one type per similar group
        """
        # Define similar type groups
        similar_groups = [
            ["simple_bar_chart", "grouped_bar_chart", "stacked_bar_chart", "horizontal_bar_chart"],
            ["simple_line_chart", "multi_line_chart", "area_chart", "stacked_area_chart"],
            ["scatter_plot", "bubble_plot"],
            ["pie_chart", "donut_chart"],
        ]
        
        selected = []
        for chart_type in types:
            # Check if any similar type already selected
            is_similar_to_selected = False
            for group in similar_groups:
                if chart_type in group:
                    if any(s in group for s in selected):
                        is_similar_to_selected = True
                        break
            
            if not is_similar_to_selected:
                selected.append(chart_type)
        
        return selected
    
    # =========================================================================
    # Main API: Generate Diverse Candidates
    # =========================================================================
    
    def generate_candidates(
        self,
        semantic_ctx: SemanticContext,
        num_initial: int = 10,
        num_final: int = 5,
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> List[FeatureCandidate]:
        """
        Generate diverse Key Feature Candidates for a semantic context.
        
        Process:
        1. LLM generates 8-10 initial candidates (divergent thinking)
        2. Diversity maximization selects top N (default 5)
        
        Args:
            semantic_ctx: Semantic context with SQL summary, schema, etc.
            num_initial: Number of initial candidates to generate
            num_final: Number of final candidates to return
            diversity_constraints: Global diversity constraints from tracker
        
        Returns:
            List of FeatureCandidate, each representing a different vis approach
        """
        # Step 1: LLM divergent generation (with diversity guidance)
        initial_candidates = self._llm_generate_candidates(
            semantic_ctx, num_initial, diversity_constraints
        )
        
        if not initial_candidates:
            logger.warning("No candidates generated, using fallback")
            return self._generate_fallback_candidates(semantic_ctx, num_final)
        
        # Step 2: Diversity maximization
        diverse_candidates = self.maximize_diversity(initial_candidates, num_final)
        
        # Step 2.5: Post-processing filter for diversity constraints
        # In early stage parallel processing, strictly avoid overrepresented chart types
        if diversity_constraints and diversity_constraints.avoid_chart_types:
            avoid_set = set(diversity_constraints.avoid_chart_types)
            # Filter out candidates with avoided chart types
            filtered_candidates = [c for c in diverse_candidates if c.chart_type not in avoid_set]
            
            # If we filtered out too many, keep some but prioritize non-avoided ones
            if len(filtered_candidates) < num_final and len(diverse_candidates) > 0:
                # Keep filtered ones first, then add back some avoided ones if needed
                remaining_needed = num_final - len(filtered_candidates)
                avoided_candidates = [c for c in diverse_candidates if c.chart_type in avoid_set]
                filtered_candidates.extend(avoided_candidates[:remaining_needed])
                logger.debug(f"Kept {len(filtered_candidates)-len([c for c in diverse_candidates if c.chart_type not in avoid_set])} avoided chart types due to insufficient alternatives")
            
            diverse_candidates = filtered_candidates[:num_final]
        
        # Step 3: Assign unique IDs and calculate key features
        question_id = semantic_ctx.instance.question_id
        for i, candidate in enumerate(diverse_candidates):
            # Format: q{question_id}_c{index}
            candidate.candidate_id = f"q{question_id}_c{i}"
            if not candidate.features:
                candidate.features = self._generate_features_for_candidate(
                    candidate, semantic_ctx
                )
        
        return diverse_candidates
    
    def _llm_generate_candidates(
        self,
        semantic_ctx: SemanticContext,
        num_candidates: int = 10,
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> List[FeatureCandidate]:
        """
        Use LLM to generate diverse visualization candidates.
        
        Uses a three-phase prompt:
        1. Understand the domain (what kind of database is this?)
        2. Discover analytical questions (what would users want to know?)
        3. Design visualizations (how to visualize each question?)
        
        Diversity constraints from global tracker are injected into the prompt
        to encourage underrepresented chart types and aggregations.
        """
        
        # Build context
        question = semantic_ctx.instance.question
        evidence = semantic_ctx.instance.evidence
        sql_summary = semantic_ctx.sql_summary
        schema_info = semantic_ctx.schema_info
        
        # Format SQL summary with execution results
        db_id = semantic_ctx.instance.db_id
        sql_summary_text = self._format_sql_summary(sql_summary, db_id) if sql_summary else ""
        
        # Format schema using table groups (FK-organized)
        schema_text = self._format_schema(schema_info) if schema_info else ""

        # Sample chart types for prompt
        config = get_config()
        selected_chart_types = self._sample_chart_types_for_prompt(
            num_types=config.pipeline.max_example_images,
            diversity_constraints=diversity_constraints
        )
        
        # Execute chart examples to generate images (if enabled)
        chart_images = self._execute_example_charts(
            selected_chart_types,
            diversity_constraints
        )
        
        # Build image reference text for prompt
        image_references = []
        if chart_images:
            for i, chart_info in enumerate(chart_images, 1):
                if chart_info['image_base64']:
                    image_references.append(
                        f"Image {i}: {chart_info['chart_type']} ({chart_info.get('category', 'N/A')})"
                    )
        
        image_reference_section = ""
        if image_references:
            image_reference_section = f"""
## Example Chart Types (Visual References)

The following chart types are shown as visual examples in the accompanying images:

{chr(10).join(image_references)}

"""
        
        # Build chart catalog text for prompt (code examples)
        chart_catalog_text = self.sample_and_format_chart_types(
            num_types=5,
            diversity_constraints=diversity_constraints
        )
        
        # Build diversity guidance from constraints
        diversity_guidance = self._build_diversity_guidance(diversity_constraints)
        
        prompt = f"""You are a senior data visualization expert creating rich, insightful visualizations.

## Context
**Original Query**: {question}
**Domain Knowledge**: {evidence or "None provided"}

## Database Schema (organized by table relationships)
{schema_text}

## Original SQL
{sql_summary_text}

## Your Task

### Phase 1: Understand the Domain
Identify the database domain (finance, education, healthcare, sports, e-commerce, etc.).
Think about who uses this data and what insights they seek.

### Phase 2: Discover Analytical Questions
Generate {num_candidates} DIVERSE analytical questions that could be visualized.
- Explore DIFFERENT analytical angles (trends, comparisons, distributions, rankings, correlations)
- Go BEYOND the original query - explore the full database
- Consider different stakeholder perspectives

### Phase 3: Design Simple, Effective Visualizations
For each question, design a visualization that focuses on clarity and the chosen chart type's core functionality.
{image_reference_section}
**Example Chart Type Code:**

{chart_catalog_text}

**CRITICAL CHART TYPE CONSTRAINTS**:
1. Each candidate MUST use EXACTLY ONE chart type from the examples above
2. DO NOT combine multiple chart types (e.g., no dual-axis charts, no bar+line combinations)
3. DO NOT use similar variants - pick ONE specific type (e.g., choose "grouped_bar_chart" OR "stacked_bar_chart", not both)
4. The chart_type field must match exactly one type name from the catalog
5. In data_mapping, do NOT use "secondary" field (no dual-axis), and "layers" should only be used if all layers are the same mark type
6. Keep visualization_features MINIMAL - only include features essential to the chosen chart type
7. DO NOT add extra advanced features like dual axes, reference lines, complex interactions, or window functions unless they are core to the chart type itself

{diversity_guidance}

For each candidate, provide:
```json
{{
  "analytical_question": "Clear analytical question being answered",
  "vis_intent": "What insight this visualization reveals",
  "vis_intent_type": "trend|comparison|distribution|ranking|correlation|composition",
  
  "chart_category": "category from list above (e.g., Bar Charts, Line Charts, Interactive Charts)",
  "chart_type": "specific chart type from category",
  
  "data_mapping": {{
    "primary": {{
      "x": "field_name (type: temporal/quantitative/nominal)",
      "y": "field_name (aggregation: sum/avg/count/etc)",
      "color": "optional_field_name"
    }},
    // Optional: for dual-axis charts
    "secondary": {{
      "y": "another_field (aggregation)"
    }},
    // Optional: for multi-layer charts
    "layers": [
      {{"type": "scatter|line|bar|rule|area", "mark_description": "description of this layer"}}
    ],
    // Optional: for faceted charts
    "facet": {{
      "field": "field_name",
      "type": "row|column|wrap"
    }}
  }},
  
  "visualization_features": [
    // ONLY list features that are ESSENTIAL for the chosen chart_type
    // DO NOT add extra advanced features beyond what the chart type requires
    // Keep this list minimal - only basic features inherent to the chart type
    // Format: "feature_type: description"
    // Examples for basic features:
    // "aggregation: sum values by category",
    // "sorting: order by value descending",
    // "time_unit: group by month"
    // DO NOT include: dual_y_axis, window_rank, reference_line, complex interactions, etc.
  ],
  
  "tables_needed": ["table1", "table2"]
}}
```

Return a JSON object with "candidates" array containing {num_candidates} candidates.
Focus on DIVERSE visualization approaches - vary chart types, data mappings, and features.
Output only the JSON object.
"""
        
        # Prepare images for multimodal input
        valid_images = [
            img_info['image_base64'] 
            for img_info in chart_images 
            if img_info.get('image_base64')
        ]
        
        # Call LLM with or without images
        if valid_images:
            logger.info(f"Generating candidates with {len(valid_images)} example chart images")
            result = self.llm.complete_with_images(
                prompt=prompt,
                images=valid_images,
                process_name="key_feature_generator.generate_candidates",
                response_format="json_object"
            )
        else:
            logger.debug("Generating candidates without images (execution disabled or all failed)")
            result = self.llm.complete_json(
                prompt,
                process_name="key_feature_generator.generate_candidates"
            )
        
        config = get_config()
        debug_mode = config.pipeline.debug_mode
        
        # Debug: log result type and content
        if not isinstance(result, dict):
            logger.error(f"LLM returned non-dict result: type={type(result)}, value={result}")
            logger.error("This might be due to thinking/reasoning output being parsed incorrectly")
            return []
        
        candidates = []
        # Check if result is a valid dict before accessing
        if result and isinstance(result, dict) and "candidates" in result:
            for i, c in enumerate(result["candidates"]):
                try:
                    intent_type_str = c.get("vis_intent_type", "comparison")
                    try:
                        intent_type = VisIntentType(intent_type_str.lower())
                    except ValueError:
                        intent_type = VisIntentType.COMPARISON
                    
                    # Handle tables_needed - could be list or string
                    tables_needed = c.get("tables_needed", [])
                    if isinstance(tables_needed, str):
                        tables_needed = [t.strip() for t in tables_needed.split(",")]
                    
                    # Get chart type and category
                    chart_type = c.get("chart_type") or "simple_bar_chart"
                    chart_category = c.get("chart_category") or self.get_category_for_type(chart_type)
                    
                    # Get new semantic fields
                    data_mapping = c.get("data_mapping", {})
                    visualization_features = c.get("visualization_features", [])
                    
                    # Extract dimensions/measures from data_mapping
                    dimensions = self._extract_dimensions_from_mapping(data_mapping)
                    measures = self._extract_measures_from_mapping(data_mapping)
                    aggregation = self._extract_aggregation_from_mapping(data_mapping)
                    
                    candidate = FeatureCandidate(
                        candidate_id=f"temp_{i}",
                        chart_type=chart_type,
                        chart_category=chart_category,
                        # Intent
                        vis_intent=c.get("vis_intent") or "",
                        vis_intent_type=intent_type,
                        analytical_question=c.get("analytical_question") or "",
                        tables_needed=tables_needed or [],
                        # New semantic fields
                        data_mapping=data_mapping,
                        visualization_features=visualization_features,
                        # Basic fields (extracted from data_mapping)
                        dimensions=dimensions,
                        measures=measures,
                        aggregation=aggregation,
                        features=[]  # Will be filled later
                    )
                    candidates.append(candidate)
                except Exception as e:
                    if debug_mode:
                        raise  # Re-raise in debug mode
                    logger.debug(f"Failed to parse candidate {i}: {e}")
                    continue
        
        # Validate that each candidate uses a single, valid chart type
        validated_candidates = []
        for candidate in candidates:
            if self._is_valid_single_chart_type(candidate):
                validated_candidates.append(candidate)
            else:
                logger.warning(f"Candidate rejected: chart_type='{candidate.chart_type}' is invalid or combined")
        
        if len(validated_candidates) < len(candidates):
            logger.info(f"Filtered out {len(candidates) - len(validated_candidates)} candidates with invalid/combined chart types")
        
        return validated_candidates
    
    def _is_valid_single_chart_type(self, candidate: FeatureCandidate) -> bool:
        """
        Validate that candidate uses a valid chart type from the catalog.
        
        Returns:
            True if chart type exists in catalog, False otherwise
        """
        # Check if chart_type exists in catalog
        if candidate.chart_type not in self.ALL_CHART_TYPES:
            logger.debug(f"Chart type '{candidate.chart_type}' not in catalog")
            return False
        
        return True
    
    # =========================================================================
    # Diversity Maximization
    # =========================================================================
    
    def maximize_diversity(
        self,
        candidates: List[FeatureCandidate],
        n: int
    ) -> List[FeatureCandidate]:
        """
        Select N most diverse candidates using greedy max-min algorithm.
        
        Algorithm:
        1. Start with first candidate
        2. Iteratively add candidate that maximizes minimum distance to selected set
        
        Diversity is measured by:
        - Chart type difference (+2.0)
        - Dimension field difference (+1.5)
        - Aggregation difference (+1.0)
        - Vis intent type difference (+1.0)
        """
        if len(candidates) <= n:
            return candidates
        
        if not candidates:
            return []
        
        # Start with first candidate
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        while len(selected) < n and remaining:
            best_candidate = None
            max_min_distance = -1
            
            for c in remaining:
                # Calculate minimum distance to any selected candidate
                min_dist = min(self._diversity_distance(c, s) for s in selected)
                
                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_candidate = c
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                best_candidate.diversity_score = max_min_distance
            else:
                break
        
        return selected
    
    def _diversity_distance(
        self,
        c1: FeatureCandidate,
        c2: FeatureCandidate
    ) -> float:
        """Calculate diversity distance between two candidates."""
        score = 0.0
        
        # Chart category difference: +2.0
        # If categories are different, it's a bigger difference than just types
        if c1.chart_category != c2.chart_category:
            score += 2.0
            
        # Chart type difference: +1.0
        # If types are different (even in same category), add score
        if c1.chart_type != c2.chart_type:
            score += 1.0
        
        # Dimension fields: less overlap = higher score
        dims1 = set(d.lower() for d in (c1.dimensions or []))
        dims2 = set(d.lower() for d in (c2.dimensions or []))
        if dims1 or dims2:
            overlap = len(dims1 & dims2)
            total = len(dims1 | dims2)
            if total > 0:
                score += (1 - overlap / total) * 1.5
        
        # Measure fields: less overlap = higher score
        meas1 = set(m.lower() for m in (c1.measures or []))
        meas2 = set(m.lower() for m in (c2.measures or []))
        if meas1 or meas2:
            overlap = len(meas1 & meas2)
            total = len(meas1 | meas2)
            if total > 0:
                score += (1 - overlap / total) * 1.0
        
        # Tables needed: less overlap = higher score (encourages cross-table exploration)
        tables1 = set(t.lower() for t in (c1.tables_needed or []))
        tables2 = set(t.lower() for t in (c2.tables_needed or []))
        if tables1 or tables2:
            overlap = len(tables1 & tables2)
            total = len(tables1 | tables2)
            if total > 0:
                score += (1 - overlap / total) * 1.5
        
        # Aggregation difference: +1.0
        agg1 = (c1.aggregation or "").upper()
        agg2 = (c2.aggregation or "").upper()
        if agg1 != agg2:
            score += 1.0
        
        # Vis intent type difference: +1.0
        if c1.vis_intent_type != c2.vis_intent_type:
            score += 1.0
        
        # ===== Semantic Features Diversity =====
        
        # Visualization features overlap: less overlap = higher score
        feat1 = set(self._parse_feature_types(c1.visualization_features))
        feat2 = set(self._parse_feature_types(c2.visualization_features))
        if feat1 or feat2:
            overlap = len(feat1 & feat2)
            total = len(feat1 | feat2)
            if total > 0:
                score += (1 - overlap / total) * 2.0  # High weight for feature diversity
        
        # Data mapping complexity difference
        complexity1 = self._estimate_mapping_complexity(c1.data_mapping)
        complexity2 = self._estimate_mapping_complexity(c2.data_mapping)
        score += abs(complexity1 - complexity2) * 0.5
        
        return score
    
    # =========================================================================
    # Key Features Generation for a Candidate
    # =========================================================================
    
    def _generate_features_for_candidate(
        self,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext
    ) -> List[Feature]:
        """Generate Vega-Lite key features for a specific candidate."""
        features = []
        
        # Mark type
        mark_type = self._chart_type_to_mark(candidate.chart_type)
        features.append(Feature(
            path="mark.type",
            op=FeatureOp.EQ,
            value=mark_type
        ))
        
        # X encoding (first dimension)
        if candidate.dimensions:
            dim_field = candidate.dimensions[0]
            features.append(Feature(
                path="encoding.x.field",
                op=FeatureOp.EQ,
                value=dim_field
            ))
            
            # Determine type
            dim_type = self._infer_field_type(dim_field, semantic_ctx)
            features.append(Feature(
                path="encoding.x.type",
                op=FeatureOp.EQ,
                value=dim_type
            ))
        
        # Y encoding (first measure with aggregation)
        if candidate.measures:
            measure_field = candidate.measures[0]
            features.append(Feature(
                path="encoding.y.field",
                op=FeatureOp.EQ,
                value=measure_field
            ))
            features.append(Feature(
                path="encoding.y.aggregate",
                op=FeatureOp.EQ,
                value=candidate.aggregation.lower()
            ))
            features.append(Feature(
                path="encoding.y.type",
                op=FeatureOp.EQ,
                value="quantitative"
            ))
        
        # Color encoding (second dimension if exists)
        if len(candidate.dimensions) > 1:
            features.append(Feature(
                path="encoding.color.field",
                op=FeatureOp.EQ,
                value=candidate.dimensions[1]
            ))
        
        # Add contract must-haves
        try:
            contract = self.contract_manager.build_contract(candidate.chart_type)
            existing_paths = {f.path for f in features}
            for must_have in contract.must_have:
                if must_have.path not in existing_paths:
                    features.append(must_have)
        except Exception as e:
            config = get_config()
            if config.pipeline.debug_mode:
                raise  # Re-raise in debug mode
            pass  # Contract may not exist for all chart types
        
        return features
    
    def _chart_type_to_mark(self, chart_type: str) -> str:
        """Convert chart type to Vega-Lite mark type."""
        chart_type_lower = chart_type.lower()
        
        if "bar" in chart_type_lower:
            return "bar"
        elif "line" in chart_type_lower:
            return "line"
        elif "scatter" in chart_type_lower or "point" in chart_type_lower:
            return "point"
        elif "area" in chart_type_lower:
            return "area"
        elif "pie" in chart_type_lower or "arc" in chart_type_lower:
            return "arc"
        elif "histogram" in chart_type_lower:
            return "bar"
        elif "box" in chart_type_lower:
            return "boxplot"
        elif "heatmap" in chart_type_lower:
            return "rect"
        else:
            return "bar"  # Default
    
    def _infer_field_type(
        self,
        field_name: str,
        semantic_ctx: SemanticContext
    ) -> str:
        """Infer Vega-Lite field type for a field."""
        field_lower = field_name.lower()
        
        # Check vis potential
        if semantic_ctx.vis_potential:
            vp = semantic_ctx.vis_potential
            if field_name in vp.temporal_fields:
                return "temporal"
            if field_name in vp.numeric_fields:
                return "quantitative"
            if field_name in vp.categorical_fields:
                return "nominal"
        
        # Heuristic based on name
        temporal_keywords = ["date", "time", "year", "month", "day", "created", "updated"]
        if any(k in field_lower for k in temporal_keywords):
            return "temporal"
        
        if field_lower.endswith("id") or field_lower.endswith("code"):
            return "nominal"
        
        return "nominal"  # Default
    
    def _extract_dimensions_from_mapping(self, data_mapping: Dict[str, Any]) -> List[str]:
        """Extract dimension fields from data_mapping structure."""
        dims = []
        primary = data_mapping.get("primary") or {}
        
        # Extract from primary encodings
        for channel in ["x", "color", "row", "column", "facet"]:
            if channel in primary:
                field_desc = primary[channel]
                # Extract field name (remove type annotations in parentheses)
                field_name = field_desc.split("(")[0].strip() if isinstance(field_desc, str) else str(field_desc)
                if field_name:
                    dims.append(field_name)
        
        # Extract from facet if present
        facet = data_mapping.get("facet") or {}
        if "field" in facet:
            field_name = facet["field"]
            if field_name and field_name not in dims:
                dims.append(field_name)
        
        return dims
    
    def _extract_measures_from_mapping(self, data_mapping: Dict[str, Any]) -> List[str]:
        """Extract measure fields from data_mapping structure."""
        measures = []
        
        # Extract from primary y encoding
        primary = data_mapping.get("primary") or {}
        if "y" in primary:
            field_desc = primary["y"]
            # Extract field name (remove aggregation in parentheses)
            field_name = field_desc.split("(")[0].strip() if isinstance(field_desc, str) else str(field_desc)
            if field_name:
                measures.append(field_name)
        
        # Extract from secondary (dual-axis)
        secondary = data_mapping.get("secondary") or {}
        if "y" in secondary:
            field_desc = secondary["y"]
            field_name = field_desc.split("(")[0].strip() if isinstance(field_desc, str) else str(field_desc)
            if field_name and field_name not in measures:
                measures.append(field_name)
        
        # Extract from size encoding if present
        if "size" in primary:
            field_desc = primary["size"]
            field_name = field_desc.split("(")[0].strip() if isinstance(field_desc, str) else str(field_desc)
            if field_name and field_name not in measures:
                measures.append(field_name)
        
        return measures
    
    def _extract_aggregation_from_mapping(self, data_mapping: Dict[str, Any]) -> str:
        """Extract aggregation type from data_mapping (from primary y encoding)."""
        primary = data_mapping.get("primary") or {}
        if "y" in primary:
            field_desc = primary["y"]
            if isinstance(field_desc, str) and "(" in field_desc:
                # Extract aggregation from parentheses, e.g., "Revenue (sum)" -> "sum"
                agg_part = field_desc.split("(")[1].split(")")[0].strip().upper()
                # Map common variations
                agg_map = {
                    "SUM": "SUM", "AVG": "AVG", "AVERAGE": "AVG", "MEAN": "AVG",
                    "COUNT": "COUNT", "MAX": "MAX", "MIN": "MIN",
                    "MEDIAN": "MEDIAN", "DISTINCT": "DISTINCT"
                }
                return agg_map.get(agg_part, "COUNT")
        
        return "COUNT"  # Default
    
    def _parse_feature_types(self, features: List[str]) -> Set[str]:
        """Extract feature types from visualization_features list."""
        types = set()
        for feat in features:
            if isinstance(feat, str) and ":" in feat:
                # Format: "type: description"
                type_part = feat.split(":")[0].strip().lower()
                types.add(type_part)
        return types
    
    def _estimate_mapping_complexity(self, mapping: Dict[str, Any]) -> int:
        """Estimate complexity of data_mapping structure."""
        complexity = 0
        
        # Dual-axis adds complexity
        if "secondary" in mapping:
            complexity += 2
        
        # Multiple layers add complexity
        if "layers" in mapping:
            layers = mapping.get("layers") or []
            complexity += len(layers)
        
        # Faceting adds complexity
        if "facet" in mapping:
            complexity += 2
        
        # Count number of encodings in primary
        primary = mapping.get("primary") or {}
        complexity += len(primary)
        
        return complexity
    
    # =========================================================================
    # Fallback Generation
    # =========================================================================
    
    def _generate_fallback_candidates(
        self,
        semantic_ctx: SemanticContext,
        num_candidates: int
    ) -> List[FeatureCandidate]:
        """Generate fallback candidates when LLM fails."""
        candidates = []
        
        # Get available fields from vis potential
        vp = semantic_ctx.vis_potential
        dims = vp.categorical_fields if vp else []
        measures = vp.numeric_fields if vp else []
        temporal = vp.temporal_fields if vp else []
        
        # If no fields available, use SQL summary
        if not dims and not measures:
            if semantic_ctx.sql_summary:
                dims = semantic_ctx.sql_summary.groupby_dims
                measures = [m.get("field", "") for m in semantic_ctx.sql_summary.measures if m.get("field")]
        
        # Default fallback
        if not dims:
            dims = ["category"]
        if not measures:
            measures = ["value"]
        
        # Generate diverse candidates with categories
        chart_configs = [
            ("simple_bar_chart", "Simple Charts", VisIntentType.COMPARISON, "SUM"),
            ("simple_line_chart", "Simple Charts", VisIntentType.TREND, "AVG"),
            ("grouped_bar_chart", "Bar Charts", VisIntentType.RANKING, "SUM"),
            ("pie_chart", "Circular Plots", VisIntentType.COMPOSITION, "SUM"),
            ("bubble_plot", "Scatter Plots", VisIntentType.CORRELATION, "COUNT"),
        ]
        
        question_id = semantic_ctx.instance.question_id
        for i, (chart_type, chart_category, intent_type, agg) in enumerate(chart_configs[:num_candidates]):
            candidate = FeatureCandidate(
                # Format: q{question_id}_c{index}
                candidate_id=f"q{question_id}_c{i}",
                chart_type=chart_type,
                chart_category=chart_category,
                dimensions=dims[:1] if dims else [],
                measures=measures[:1] if measures else [],
                aggregation=agg,
                vis_intent=f"Show {intent_type.value} of data",
                vis_intent_type=intent_type,
                features=[]
            )
            candidate.features = self._generate_features_for_candidate(
                candidate, semantic_ctx
            )
            candidates.append(candidate)
        
        return candidates
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_diversity_guidance(self, constraints: Optional[DiversityConstraints]) -> str:
        """
        Build diversity guidance text for LLM prompt based on global constraints.
        
        This guides the LLM to generate candidates that improve overall dataset diversity
        by preferring underrepresented chart types, categories, aggregations, etc.
        
        Provides MULTIPLE recommendations across different dimensions.
        """
        if not constraints:
            return ""
        
        sections = []
        
        # =========================================================================
        # Section 1: Chart Category & Type Diversity
        # =========================================================================
        chart_lines = []
        
        # Preferred chart categories (derive from preferred types)
        if constraints.preferred_chart_types:
            preferred_categories = set()
            # Randomly sample from preferred types to increase diversity
            sampled_types = random.sample(
                constraints.preferred_chart_types,
                min(10, len(constraints.preferred_chart_types))
            ) if len(constraints.preferred_chart_types) > 10 else constraints.preferred_chart_types
            for ct in sampled_types:
                cat = self.get_category_for_type(ct)
                if cat != "Other":
                    preferred_categories.add(cat)
            if preferred_categories:
                # Shuffle categories for variety
                categories_list = list(preferred_categories)
                random.shuffle(categories_list)
                chart_lines.append(f"  - Prioritize these CATEGORIES: {', '.join(categories_list[:5])}")
            
            # Also show specific types (randomly sampled and shuffled)
            preferred_types = random.sample(
                constraints.preferred_chart_types,
                min(8, len(constraints.preferred_chart_types))
            ) if len(constraints.preferred_chart_types) > 8 else constraints.preferred_chart_types
            random.shuffle(preferred_types)
            chart_lines.append(f"  - Underrepresented chart types: {', '.join(preferred_types)}")
        
        if constraints.avoid_chart_types:
            avoid = random.sample(
                constraints.avoid_chart_types,
                min(5, len(constraints.avoid_chart_types))
            ) if len(constraints.avoid_chart_types) > 5 else constraints.avoid_chart_types
            random.shuffle(avoid)
            chart_lines.append(f"  - Avoid overused types: {', '.join(avoid)}")
        
        if constraints.preferred_chart_categories:
            cats = random.sample(
                constraints.preferred_chart_categories,
                min(4, len(constraints.preferred_chart_categories))
            ) if len(constraints.preferred_chart_categories) > 4 else constraints.preferred_chart_categories
            random.shuffle(cats)
            chart_lines.append(f"  - Focus on categories: {', '.join(cats)}")
        
        if chart_lines:
            sections.append("**CHART DIVERSITY** (vary across candidates):\n" + "\n".join(chart_lines))
        
        # =========================================================================
        # Section 2: Aggregation Diversity
        # =========================================================================
        agg_lines = []
        
        if constraints.preferred_aggregations:
            aggs = random.sample(
                constraints.preferred_aggregations,
                min(5, len(constraints.preferred_aggregations))
            ) if len(constraints.preferred_aggregations) > 5 else constraints.preferred_aggregations
            random.shuffle(aggs)
            agg_lines.append(f"  - Use more of: {', '.join(aggs)}")
        
        if constraints.avoid_aggregations:
            avoid_aggs = random.sample(
                constraints.avoid_aggregations,
                min(3, len(constraints.avoid_aggregations))
            ) if len(constraints.avoid_aggregations) > 3 else constraints.avoid_aggregations
            random.shuffle(avoid_aggs)
            agg_lines.append(f"  - Use less of: {', '.join(avoid_aggs)}")
        
        if agg_lines:
            sections.append("**AGGREGATION DIVERSITY**:\n" + "\n".join(agg_lines))
        
        # =========================================================================
        # Section 3: Visual Effects Diversity (Transforms, Interactions, Composition)
        # =========================================================================
        effect_lines = []
        
        if constraints.preferred_transforms:
            transforms = random.sample(
                constraints.preferred_transforms,
                min(4, len(constraints.preferred_transforms))
            ) if len(constraints.preferred_transforms) > 4 else constraints.preferred_transforms
            random.shuffle(transforms)
            effect_lines.append(f"  - Try these transforms: {', '.join(transforms)}")
        
        if constraints.preferred_composition:
            effect_lines.append(f"  - Try '{constraints.preferred_composition}' composition")
        
        if constraints.preferred_interaction:
            effect_lines.append(f"  - Add '{constraints.preferred_interaction}' interaction")
        
        if effect_lines:
            sections.append("**VISUAL EFFECTS DIVERSITY** (add to some candidates):\n" + "\n".join(effect_lines))
        
        # =========================================================================
        # Section 4: Time Unit Diversity
        # =========================================================================
        if constraints.preferred_time_units:
            time_units = random.sample(
                constraints.preferred_time_units,
                min(4, len(constraints.preferred_time_units))
            ) if len(constraints.preferred_time_units) > 4 else constraints.preferred_time_units
            random.shuffle(time_units)
            sections.append(f"**TIME GRANULARITY**: For temporal data, try: {', '.join(time_units)}")
        
        # =========================================================================
        # Section 5: Encoding Pattern Hints
        # =========================================================================
        encoding_lines = []
        
        if constraints.preferred_encoding_pattern:
            encoding_lines.append(f"  - Try encoding pattern: {constraints.preferred_encoding_pattern}")
        
        if constraints.avoid_encoding_patterns:
            avoid_patterns = random.sample(
                constraints.avoid_encoding_patterns,
                min(3, len(constraints.avoid_encoding_patterns))
            ) if len(constraints.avoid_encoding_patterns) > 3 else constraints.avoid_encoding_patterns
            random.shuffle(avoid_patterns)
            encoding_lines.append(f"  - Avoid patterns: {', '.join(avoid_patterns)}")
        
        if encoding_lines:
            sections.append("**ENCODING DIVERSITY**:\n" + "\n".join(encoding_lines))
        
        # =========================================================================
        # Combine all sections
        # =========================================================================
        if not sections:
            return ""
        
        guidance = "\n\n### Diversity Guidance (improve dataset variety)\n" + "\n\n".join(sections) + "\n"
        return guidance
    
    def _format_sql_summary(self, sql_summary: SQLSemanticSummary, db_id: str) -> str:
        """
        Format SQL summary with execution results as markdown table.
        
        Args:
            sql_summary: SQL semantic summary
            db_id: Database ID for execution
            
        Returns:
            Formatted markdown with SQL code block and execution result table
        """
        # Format SQL in code block
        sql_text = f"```sql\n{sql_summary.sql}\n```"
        
        # Execute SQL and get results
        try:
            sqlite_client = get_sqlite_client()
            df = sqlite_client.execute_safe(db_id, sql_summary.sql, limit=10)
            
            if df is not None and not df.empty:
                # Generate markdown table
                table_text = self._df_to_markdown_table(df.head(5), full_df=df)
                
                # Add row count info
                total_rows = len(df)
                if total_rows > 5:
                    table_text += f"\n... +{total_rows - 5} more rows"
                
                result = f"{sql_text}\n\n## Data Sample (first 5 rows of total {total_rows} rows, with distinct counts per column)\n{table_text}"
            else:
                result = f"{sql_text}\n\n*(SQL execution returned no results)*"
                
        except Exception as e:
            logger.warning(f"Failed to execute SQL for summary: {e}")
            result = f"{sql_text}\n\n*(SQL execution failed: {e})*"
        
        return result
    
    def _df_to_markdown_table(self, df: pd.DataFrame, max_cols: int = 10, full_df: pd.DataFrame = None) -> str:
        """
        Convert DataFrame to markdown table format with column statistics.
        
        Args:
            df: DataFrame to display (usually a head() subset)
            max_cols: Maximum columns to display
            full_df: Full DataFrame for computing distinct counts (if None, uses df)
        """
        if df.empty:
            return "*(empty result)*"
        
        # Use full_df for statistics if provided
        stats_df = full_df if full_df is not None else df
        
        # Limit columns for display
        display_df = df.iloc[:, :max_cols] if len(df.columns) > max_cols else df
        stats_display_df = stats_df.iloc[:, :max_cols] if len(stats_df.columns) > max_cols else stats_df
        
        # Build header
        headers = list(display_df.columns)
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        
        # Build distinct count row
        distinct_counts = []
        for col in display_df.columns:
            distinct_count = stats_display_df[col].nunique()
            distinct_counts.append(f"*{distinct_count} distinct*")
        distinct_row = "| " + " | ".join(distinct_counts) + " |"
        
        # Build data rows
        data_rows = []
        for _, row in display_df.iterrows():
            values = []
            for val in row:
                # Truncate long values
                str_val = str(val) if pd.notna(val) else "NULL"
                if len(str_val) > 30:
                    str_val = str_val[:27] + "..."
                values.append(str_val)
            data_rows.append("| " + " | ".join(values) + " |")
        
        result = "\n".join([header_row, separator, distinct_row] + data_rows)
        
        if len(df.columns) > max_cols:
            result += f"\n\n*(showing {max_cols} of {len(df.columns)} columns)*"
        
        return result
    
    def _format_schema(self, schema_info: DatabaseInfo) -> str:
        """
        Format schema for prompt using table groups with sample data.
        
        Tables are organized by FK relationships, and each table includes
        sample data to help LLM understand the data content.
        """
        if not schema_info:
            return "No schema available"
        
        # Use schema_processor's format with groups and sample data
        schema_processor = get_schema_processor()
        return schema_processor.format_schema_with_groups_and_samples(schema_info)
    
    def _format_vis_potential(self, vis_potential: VisPotential) -> str:
        """Format vis potential for prompt."""
        if not vis_potential:
            return "No vis potential analysis available"
        
        parts = []
        if vis_potential.temporal_fields:
            parts.append(f"Temporal fields: {', '.join(vis_potential.temporal_fields)}")
        if vis_potential.categorical_fields:
            parts.append(f"Categorical fields: {', '.join(vis_potential.categorical_fields)}")
        if vis_potential.numeric_fields:
            parts.append(f"Numeric fields: {', '.join(vis_potential.numeric_fields)}")
        # Do NOT include suggested chart types here; it anchors the model.
        parts.append("Hint: Identifier fields (ending with 'id') are good COUNT/COUNT(DISTINCT) measures.")
        
        return "\n".join(parts) if parts else "No fields identified"
    
    # =========================================================================
    # Utility: Injectable Pool (for ambiguity injection)
    # =========================================================================
    
    def get_injectable_paths(self, candidate: FeatureCandidate) -> List[str]:
        """Get paths that can be made ambiguous for a candidate."""
        injectable = []
        
        # Aggregation is often injectable
        injectable.append("encoding.y.aggregate")
        
        # Time unit for temporal fields
        if any("date" in d.lower() or "time" in d.lower() for d in candidate.dimensions):
            injectable.append("encoding.x.timeUnit")
        
        # Color field
        injectable.append("encoding.color.field")
        
        # Sort order
        injectable.append("encoding.x.sort")
        injectable.append("encoding.y.sort")
        
        # Mark type (if alternatives exist)
        if candidate.chart_type in ["bar_chart", "line_chart", "area_chart"]:
            injectable.append("mark.type")
        
        return injectable


# Singleton instance
_generator: Optional[KeyFeatureGenerator] = None


def get_key_feature_generator() -> KeyFeatureGenerator:
    """Get or create singleton key feature generator."""
    global _generator
    if _generator is None:
        _generator = KeyFeatureGenerator()
    return _generator
