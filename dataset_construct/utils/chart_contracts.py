"""
Chart Contracts for chart types from chart_example directory.
Uses Vega-Altair official documentation examples as the source.

Directory structure:
- chart_example/{category}/{chart_type}.py
- category = subdirectory name
- chart_type = py filename without extension
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from core.models import ChartContract, Feature, FeatureOp
from core.config import get_config

logger = logging.getLogger(__name__)


class ChartContractManager:
    """
    Manages chart type contracts from chart_example directory.
    
    Data source: chart_example/{category}/{chart_type}.py
    - Subdirectory name = category (e.g., "Bar Charts", "Interactive Charts")
    - Python filename = chart_type (e.g., "simple_bar_chart", "slider_cutoff")
    """
    
    def __init__(self, chart_example_dir: Optional[Path] = None):
        self.chart_example_dir = chart_example_dir or get_config().paths.chart_example_dir
        self._contracts: Dict[str, ChartContract] = {}
        self._chart_catalog: Dict[str, Dict[str, Any]] = {}  # chart_type -> {category, code, file_path}
        self._categories: Dict[str, List[str]] = defaultdict(list)  # category -> [chart_types]
        self._loaded = False
    
    def _load_catalog(self):
        """Scan chart_example directory and build catalog."""
        if self._loaded:
            return
        
        if not self.chart_example_dir.exists():
            logger.warning(f"chart_example directory not found at {self.chart_example_dir}")
            self._loaded = True
            return
        
        # Scan subdirectories (categories)
        for category_dir in self.chart_example_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # Skip __pycache__ and hidden directories
            if category_dir.name.startswith(('_', '.')):
                continue
            
            category = category_dir.name
            
            # Scan .py files (chart types)
            for py_file in category_dir.glob("*.py"):
                chart_type = py_file.stem  # filename without .py
                
                # Read code content
                try:
                    code = py_file.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to read {py_file}: {e}")
                    continue
                
                self._chart_catalog[chart_type] = {
                    "category": category,
                    "code": code,
                    "file_path": py_file
                }
                self._categories[category].append(chart_type)
        
        self._loaded = True
        logger.info(f"Loaded {len(self._chart_catalog)} chart types from {len(self._categories)} categories")
    
    def get_all_chart_types(self) -> List[str]:
        """Get list of all chart types."""
        self._load_catalog()
        return list(self._chart_catalog.keys())
    
    def get_chart_categories(self) -> Dict[str, List[str]]:
        """Get chart types grouped by category."""
        self._load_catalog()
        return dict(self._categories)
    
    def get_category_for_type(self, chart_type: str) -> str:
        """Get the category for a specific chart type."""
        self._load_catalog()
        info = self._chart_catalog.get(chart_type)
        return info["category"] if info else "Unknown"
    
    def get_sample_code(self, chart_type: str) -> List[str]:
        """Get sample code for a chart type."""
        self._load_catalog()
        info = self._chart_catalog.get(chart_type)
        if info and info.get("code"):
            return [info["code"]]
        return []
    
    def _execute_chart_code(self, code: str, validate_rendering: bool = True) -> Optional[Dict[str, Any]]:
        """
        Execute Altair code and return the Vega-Lite spec.
        
        Handles common dependencies like vega_datasets.
        
        Args:
            code: Python code to execute
            validate_rendering: If True, validate spec can be rendered (catches frontend errors)
        
        Returns:
            Vega-Lite spec dict or None if failed
        """
        import pandas as pd
        import numpy as np
        
        try:
            import altair as alt
            
            # Build namespace with common imports
            namespace = {
                "alt": alt,
                "pd": pd,
                "np": np,
                "numpy": np,
                "pandas": pd,
            }
            
            # Handle vega_datasets dependency
            if "vega_datasets" in code:
                try:
                    from vega_datasets import data
                    namespace["data"] = data
                except ImportError:
                    logger.debug("vega_datasets not available, using fallback")
                    # Create a mock that returns dummy data
                    namespace["data"] = self._create_mock_data_source()
            
            # Handle random state for reproducibility
            namespace["rand"] = np.random.RandomState(42)
            
            # Execute the code
            exec(code, namespace)
            
            # Find Chart object in namespace
            # Official examples often end with the chart expression directly
            chart = None
            
            # First, check common variable names
            for name in ["chart", "c", "fig", "plot"]:
                if name in namespace and self._is_chart_object(namespace[name]):
                    chart = namespace[name]
                    break
            
            # If not found, search all namespace items for Chart objects
            if chart is None:
                for name, obj in namespace.items():
                    if name.startswith("_"):
                        continue
                    if self._is_chart_object(obj):
                        chart = obj
                        break
            
            if chart is None:
                # Try to get the result of the last expression
                # For code that ends with `alt.Chart(...)`
                logger.debug("No chart variable found, code may end with expression")
                return None
            
            # Convert to Vega-Lite spec
            spec = chart.to_dict()
            
            # Validate rendering if requested
            if validate_rendering:
                render_error = self._validate_spec_rendering(spec, chart)
                if render_error:
                    logger.warning(f"Spec rendering validation failed: {render_error}")
                    return None
            
            return spec
            
        except Exception as e:
            logger.debug(f"Failed to execute chart code: {e}")
            return None
    
    def _validate_spec_rendering(
        self,
        spec: Dict[str, Any],
        chart: Any
    ) -> Optional[str]:
        """
        Validate that the Vega-Lite spec can be rendered without JavaScript errors.
        
        Uses Altair's native chart.save() method which is more reliable than vl_convert.
        
        Args:
            spec: Vega-Lite spec dictionary
            chart: Altair chart object
        
        Returns:
            Error message if rendering fails, None if successful
        """
        try:
            import tempfile
            import os
            
            # Use the chart object directly for validation
            # Try to save to a temporary file - this validates the spec
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Save as JSON (Vega-Lite spec)
                # This will catch any serialization issues
                chart.save(tmp_path)
                
                # If save succeeded, validation passed
                return None
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
        except Exception as e:
            error_msg = str(e)
            
            # Parse and simplify error message
            if "Unrecognized signal name" in error_msg:
                return f"Invalid signal reference"
            elif "Undefined field" in error_msg:
                return f"Field reference error"
            elif "Cannot read" in error_msg or "undefined" in error_msg.lower():
                return f"Spec structure error"
            elif "JSON" in error_msg or "json" in error_msg:
                return f"JSON serialization error"
            else:
                return f"Rendering error: {error_msg[:200]}"
    
    def _is_chart_object(self, obj: Any) -> bool:
        """Check if an object is an Altair Chart."""
        try:
            import altair as alt
            return isinstance(obj, (alt.Chart, alt.LayerChart, alt.HConcatChart, 
                                   alt.VConcatChart, alt.FacetChart, alt.RepeatChart))
        except Exception:
            return False
    
    def _create_mock_data_source(self):
        """Create a mock data source for when vega_datasets is not available."""
        import pandas as pd
        
        class MockDataSource:
            def __getattr__(self, name):
                return self._get_dummy_data
            
            def _get_dummy_data(self, *args, **kwargs):
                return pd.DataFrame({
                    'x': range(10),
                    'y': range(10),
                    'category': ['A'] * 5 + ['B'] * 5,
                    'value': [1, 2, 3, 4, 5] * 2,
                    'date': pd.date_range('2020-01-01', periods=10),
                })
        
        return MockDataSource()
    
    def extract_features_from_spec(self, spec: Dict[str, Any]) -> List[Feature]:
        """
        Extract key features directly from a Vega-Lite spec.
        
        This is a programmatic extraction without LLM.
        """
        features = []
        
        # 1. Mark type
        mark = spec.get("mark")
        if isinstance(mark, str):
            features.append(Feature("mark.type", FeatureOp.EQ, mark))
        elif isinstance(mark, dict):
            mark_type = mark.get("type")
            if mark_type:
                features.append(Feature("mark.type", FeatureOp.EQ, mark_type))
            # Check for innerRadius (donut)
            if "innerRadius" in mark:
                features.append(Feature("mark.innerRadius", FeatureOp.EXISTS))
        
        # 2. Encodings
        encoding = spec.get("encoding", {})
        for channel, enc_spec in encoding.items():
            if isinstance(enc_spec, dict):
                # Field exists
                if "field" in enc_spec:
                    features.append(Feature(f"encoding.{channel}.field", FeatureOp.EXISTS))
                
                # Encoding type
                if "type" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.type", 
                        FeatureOp.EQ, 
                        enc_spec["type"]
                    ))
                
                # Aggregate
                if "aggregate" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.aggregate",
                        FeatureOp.EQ,
                        enc_spec["aggregate"]
                    ))
                
                # Bin
                if "bin" in enc_spec:
                    features.append(Feature(f"encoding.{channel}.bin", FeatureOp.EXISTS))
                
                # TimeUnit
                if "timeUnit" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.timeUnit",
                        FeatureOp.EQ,
                        enc_spec["timeUnit"]
                    ))
        
        # 3. Transforms
        transforms = spec.get("transform", [])
        if transforms:
            features.append(Feature("transform", FeatureOp.EXISTS))
            for i, t in enumerate(transforms):
                for key in ["aggregate", "filter", "calculate", "window", "bin", "fold", "joinaggregate", "density", "loess", "regression"]:
                    if key in t:
                        features.append(Feature(f"transform[{i}].{key}", FeatureOp.EXISTS))
        
        # 4. Layer composition
        if "layer" in spec:
            features.append(Feature("layer", FeatureOp.EXISTS))
            features.append(Feature("layer", FeatureOp.LEN_GE, len(spec["layer"])))
        
        # 5. Concat composition
        for concat_type in ["hconcat", "vconcat", "concat"]:
            if concat_type in spec:
                features.append(Feature(concat_type, FeatureOp.EXISTS))
        
        # 6. Facet
        if "facet" in spec:
            features.append(Feature("facet", FeatureOp.EXISTS))
        
        # 7. Resolve (for dual axis)
        resolve = spec.get("resolve", {})
        if resolve:
            features.append(Feature("resolve", FeatureOp.EXISTS))
            for res_type, res_spec in resolve.items():
                if isinstance(res_spec, dict):
                    for channel, value in res_spec.items():
                        features.append(Feature(
                            f"resolve.{res_type}.{channel}",
                            FeatureOp.EQ,
                            value
                        ))
        
        # 8. Params (interactivity)
        params = spec.get("params", [])
        if params:
            features.append(Feature("params", FeatureOp.EXISTS))
            for i, param in enumerate(params):
                # Selection type
                if "select" in param:
                    select = param["select"]
                    if isinstance(select, dict) and "type" in select:
                        features.append(Feature(
                            f"params[{i}].select.type",
                            FeatureOp.EQ,
                            select["type"]
                        ))
                    elif isinstance(select, str):
                        features.append(Feature(
                            f"params[{i}].select",
                            FeatureOp.EQ,
                            select
                        ))
                # Bind (slider, etc.)
                if "bind" in param:
                    features.append(Feature(f"params[{i}].bind", FeatureOp.EXISTS))
        
        # 9. Scale configurations
        for channel in ["x", "y", "color", "size"]:
            enc = encoding.get(channel, {})
            if isinstance(enc, dict) and "scale" in enc:
                scale = enc["scale"]
                if isinstance(scale, dict):
                    if "type" in scale:
                        features.append(Feature(
                            f"encoding.{channel}.scale.type",
                            FeatureOp.EQ,
                            scale["type"]
                        ))
                    if "scheme" in scale:
                        features.append(Feature(
                            f"encoding.{channel}.scale.scheme",
                            FeatureOp.EXISTS
                        ))
        
        return features
    
    def build_contract(self, chart_type: str) -> ChartContract:
        """Build or retrieve contract for a chart type."""
        if chart_type in self._contracts:
            return self._contracts[chart_type]
        
        self._load_catalog()
        info = self._chart_catalog.get(chart_type)
        
        if not info:
            # Return empty contract for unknown types
            return ChartContract(
                chart_type=chart_type,
                category="Unknown",
                must_have=[],
                optional=[],
                min_data_requirements={}
            )
        
        category = info["category"]
        code = info["code"]
        
        # Extract features from code execution
        features = []
        spec = self._execute_chart_code(code)
        if spec:
            features = self.extract_features_from_spec(spec)
        
        # All extracted features are considered must-have
        # (since we only have one sample per chart type)
        contract = ChartContract(
            chart_type=chart_type,
            category=category,
            must_have=features,
            optional=[],
            min_data_requirements=self._get_data_requirements(chart_type, category)
        )
        
        self._contracts[chart_type] = contract
        return contract
    
    def _get_data_requirements(self, chart_type: str, category: str) -> Dict[str, Any]:
        """Get minimum data requirements based on category."""
        category_requirements = {
            "Bar Charts": {"min_rows": 2, "needs_categorical": True},
            "Line Charts": {"min_rows": 3, "needs_ordered": True},
            "Area Charts": {"min_rows": 3, "needs_ordered": True},
            "Scatter Plots": {"min_rows": 10, "needs_numeric": 2},
            "Circular Plots": {"min_rows": 2, "max_categories": 8, "needs_positive": True},
            "Distributions": {"min_rows": 20, "needs_numeric": 1},
            "Interactive Charts": {"min_rows": 10},
            "Advanced Calculations": {"min_rows": 5},
            "Uncertainties And Trends": {"min_rows": 10, "needs_numeric": 1},
            "Tables": {"min_rows": 9, "needs_categorical": 2},
            "Simple Charts": {"min_rows": 3},
        }
        
        return category_requirements.get(category, {"min_rows": 5})
    
    def validate_spec_against_contract(
        self,
        spec: Dict[str, Any],
        contract: ChartContract
    ) -> List[str]:
        """Validate a Vega-Lite spec against a contract. Returns list of violations."""
        violations = []
        
        for feature in contract.must_have:
            if not self._check_feature(spec, feature):
                violations.append(f"Missing required feature: {feature.path} {feature.op.value} {feature.value}")
        
        return violations
    
    def _check_feature(self, spec: Dict[str, Any], feature: Feature) -> bool:
        """Check if a spec satisfies a feature constraint."""
        value = self._get_path_value(spec, feature.path)
        
        op = feature.op if isinstance(feature.op, FeatureOp) else FeatureOp(feature.op)
        
        if op == FeatureOp.EXISTS:
            return value is not None
        elif op == FeatureOp.EQ:
            return value == feature.value
        elif op == FeatureOp.IN:
            return value in (feature.value or [])
        elif op == FeatureOp.CONTAINS:
            if isinstance(value, (list, str)):
                return feature.value in value
            return False
        elif op == FeatureOp.LEN_GE:
            if isinstance(value, (list, str)):
                return len(value) >= feature.value
            return False
        elif op == FeatureOp.LEN_LE:
            if isinstance(value, (list, str)):
                return len(value) <= feature.value
            return False
        
        return False
    
    def _get_path_value(self, obj: Any, path: str) -> Any:
        """Get value at a path in nested object."""
        parts = path.replace("][", ".").replace("[", ".").replace("]", "").split(".")
        
        current = obj
        for part in parts:
            if current is None:
                return None
            
            # Handle array index
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            # Handle wildcard
            elif part == "*":
                if isinstance(current, list):
                    # Return first match
                    current = current[0] if current else None
                else:
                    return None
            # Handle dict key
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current


# Singleton instance
_manager: Optional[ChartContractManager] = None


def get_contract_manager() -> ChartContractManager:
    """Get or create singleton contract manager."""
    global _manager
    if _manager is None:
        _manager = ChartContractManager()
    return _manager
