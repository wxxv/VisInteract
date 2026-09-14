"""
Chart Catalog Loader for Vis-Interact Dataset Construction.

Dynamically loads chart types and example code from the chart_example directory.
Provides caching for performance and ensures catalog stays in sync with actual files.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ChartCatalogLoader:
    """
    Loads and manages the chart catalog from the chart_example directory.
    
    The catalog is organized by categories (subdirectories) and chart types
    (Python filenames without .py extension). Example code is extracted from
    each Python file for LLM context.
    """
    
    def __init__(self, chart_example_dir: Path):
        """
        Initialize the chart catalog loader.
        
        Args:
            chart_example_dir: Path to the chart_example directory
        """
        self.chart_example_dir = Path(chart_example_dir)
        self._catalog: Optional[Dict[str, List[str]]] = None
        self._examples: Dict[str, str] = {}  # chart_type -> example code
        self._type_to_category: Dict[str, str] = {}  # chart_type -> category
        self._type_to_path: Dict[str, Path] = {}  # chart_type -> file path
        
    def load_catalog(self) -> Dict[str, List[str]]:
        """
        Scan chart_example directory and build catalog.
        
        Returns:
            Dictionary mapping category names to lists of chart type names.
            Example: {"Simple Charts": ["simple_bar_chart", "simple_line_chart"], ...}
        """
        if self._catalog is not None:
            return self._catalog
        
        if not self.chart_example_dir.exists():
            logger.warning(f"Chart example directory does not exist: {self.chart_example_dir}")
            return {}
        
        catalog = {}
        
        # Scan subdirectories (categories)
        for category_dir in sorted(self.chart_example_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            chart_types = []
            
            # Scan Python files in category
            for py_file in sorted(category_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue  # Skip __init__.py and private files
                
                chart_type = py_file.stem  # Filename without .py
                chart_types.append(chart_type)
                
                # Build reverse mappings
                self._type_to_category[chart_type] = category_name
                self._type_to_path[chart_type] = py_file
            
            if chart_types:
                catalog[category_name] = chart_types
        
        self._catalog = catalog
        logger.info(f"Loaded chart catalog: {len(catalog)} categories, "
                   f"{sum(len(types) for types in catalog.values())} chart types")
        
        return catalog
    
    def get_example_code(self, chart_type: str) -> Optional[str]:
        """
        Get full example code for a chart type.
        
        Args:
            chart_type: The chart type name (e.g., "simple_bar_chart")
            
        Returns:
            Python code as string, or None if not found
        """
        # Check cache first
        if chart_type in self._examples:
            return self._examples[chart_type]
        
        # Ensure catalog is loaded
        if self._catalog is None:
            self.load_catalog()
        
        # Get file path
        file_path = self._type_to_path.get(chart_type)
        if file_path is None or not file_path.exists():
            logger.warning(f"Chart type not found: {chart_type}")
            return None
        
        # Read example code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Cache it
            self._examples[chart_type] = code
            return code
            
        except Exception as e:
            logger.error(f"Failed to read example code for {chart_type}: {e}")
            return None
    
    def get_category_for_type(self, chart_type: str) -> Optional[str]:
        """
        Get the category for a chart type.
        
        Args:
            chart_type: The chart type name
            
        Returns:
            Category name, or None if not found
        """
        if self._catalog is None:
            self.load_catalog()
        
        return self._type_to_category.get(chart_type)
    
    def get_chart_info(self, chart_type: str) -> Optional[Dict]:
        """
        Get structured information about a chart type.
        
        Args:
            chart_type: The chart type name
            
        Returns:
            Dictionary with keys: chart_type, category, example_code, file_path
        """
        if self._catalog is None:
            self.load_catalog()
        
        category = self.get_category_for_type(chart_type)
        if category is None:
            return None
        
        example_code = self.get_example_code(chart_type)
        file_path = self._type_to_path.get(chart_type)
        
        return {
            "chart_type": chart_type,
            "category": category,
            "example_code": example_code,
            "file_path": str(file_path) if file_path else None,
        }
    
    def get_all_chart_types(self) -> List[str]:
        """
        Get a flat list of all chart types across all categories.
        
        Returns:
            List of chart type names
        """
        if self._catalog is None:
            self.load_catalog()
        
        return [ct for types in self._catalog.values() for ct in types]
    
    def get_types_for_category(self, category: str) -> List[str]:
        """
        Get all chart types in a specific category.
        
        Args:
            category: The category name
            
        Returns:
            List of chart type names in that category
        """
        if self._catalog is None:
            self.load_catalog()
        
        return self._catalog.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """
        Get all category names.
        
        Returns:
            List of category names
        """
        if self._catalog is None:
            self.load_catalog()
        
        return list(self._catalog.keys())
    
    def format_example_with_metadata(self, chart_type: str) -> Optional[str]:
        """
        Format example code with metadata for LLM prompt.
        
        Returns a formatted string like:
        ### Chart Type: simple_bar_chart (Simple Charts category)
        
        ```python
        import altair as alt
        ...
        ```
        
        Args:
            chart_type: The chart type name
            
        Returns:
            Formatted string with chart info and example code
        """
        info = self.get_chart_info(chart_type)
        if info is None or info["example_code"] is None:
            return None
        
        category = info["category"]
        code = info["example_code"].strip()
        
        formatted = f"""### Chart Type: {chart_type} ({category} category)

```python
{code}
```"""
        
        return formatted


# Singleton instance
_catalog_loader: Optional[ChartCatalogLoader] = None


def get_chart_catalog_loader(chart_example_dir: Optional[Path] = None) -> ChartCatalogLoader:
    """
    Get or create singleton chart catalog loader.
    
    Args:
        chart_example_dir: Path to chart_example directory (only used on first call)
        
    Returns:
        ChartCatalogLoader instance
    """
    global _catalog_loader
    
    if _catalog_loader is None:
        if chart_example_dir is None:
            # Get from config
            from core.config import get_config
            config = get_config()
            chart_example_dir = config.paths.chart_example_dir
        
        _catalog_loader = ChartCatalogLoader(chart_example_dir)
        _catalog_loader.load_catalog()  # Load immediately
    
    return _catalog_loader
