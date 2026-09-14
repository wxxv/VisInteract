"""
Chart Executor for executing demonstration chart code and generating images.

Executes Altair example code in isolated namespace and renders charts to PNG images.
"""
import logging
import base64
import io
import tempfile
import signal
import ast
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when chart execution times out."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Chart execution timed out")


class ChartExecutor:
    """
    Executes demonstration chart code and generates PNG images.
    
    Handles:
    - Execution in isolated namespace
    - External dependencies (vega_datasets, pandas, numpy, altair)
    - Chart rendering to PNG
    - Base64 encoding
    - Timeout protection
    - Error handling
    """
    
    def __init__(self, timeout: float = 5.0, scale: float = 1.5):
        """
        Initialize ChartExecutor.
        
        Args:
            timeout: Maximum execution time per chart (seconds)
            scale: Scale factor for rendered images
        """
        self.timeout = timeout
        self.scale = scale
    
    def execute_chart_example(
        self,
        chart_type: str,
        example_code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute example chart code and return base64 image.
        
        Args:
            chart_type: Name of the chart type (for logging)
            example_code: Python code to execute
            
        Returns:
            Tuple of (image_base64, error_message)
            - image_base64: Base64-encoded PNG if successful, None if failed
            - error_message: Error description if failed, None if successful
        """
        try:
            # Set timeout alarm (Unix only, Windows will skip this)
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(self.timeout))
            except (AttributeError, ValueError):
                # Windows doesn't support SIGALRM, skip timeout protection
                pass
            
            # Execute code in isolated namespace with jupyter-style last expression support
            namespace = self._prepare_namespace()
            import altair as alt
            
            last_expr_value = None
            try:
                # Clean up code (remove .show(), .display(), etc.)
                modified_code = example_code.replace(".show()", "").replace(".display(", "# .display(")
                
                # Try to parse and handle last expression separately (jupyter style)
                try:
                    parsed_code = ast.parse(modified_code)
                    if parsed_code.body and isinstance(parsed_code.body[-1], ast.Expr):
                        # Last line is an expression - execute separately to capture return value
                        last_expr = ast.unparse(parsed_code.body[-1])
                        code_without_last = ast.unparse(ast.Module(body=parsed_code.body[:-1], type_ignores=[]))
                        
                        # Execute everything except last line
                        if code_without_last.strip():
                            exec(code_without_last, namespace)
                        
                        # Evaluate last expression
                        last_expr_value = eval(last_expr, namespace)
                    else:
                        # No expression at end, just execute normally
                        exec(modified_code, namespace)
                except SyntaxError:
                    # If parsing fails, fall back to simple execution
                    exec(modified_code, namespace)
                    
            except Exception as e:
                error_msg = f"Code execution failed: {type(e).__name__}: {str(e)}"
                logger.warning(f"Failed to execute {chart_type}: {error_msg}")
                return None, error_msg
            finally:
                # Cancel alarm
                try:
                    signal.alarm(0)
                except AttributeError:
                    pass
            
            # Extract chart object with multiple strategies
            chart = None
            
            # Strategy 1: Check if last expression is a chart (jupyter style)
            if last_expr_value is not None:
                if isinstance(last_expr_value, alt.TopLevelMixin):
                    chart = last_expr_value
                    logger.debug(f"Found chart from last expression (jupyter style)")
            
            # Strategy 2: Check for 'chart' variable
            if chart is None and 'chart' in namespace:
                if isinstance(namespace['chart'], alt.TopLevelMixin):
                    chart = namespace['chart']
                    logger.debug(f"Found chart from 'chart' variable")
            
            # Strategy 3: Try alternative variable names
            if chart is None:
                for var_name in ['c', 'fig', 'plot', 'visualization', 'base', 'final']:
                    if var_name in namespace and isinstance(namespace[var_name], alt.TopLevelMixin):
                        chart = namespace[var_name]
                        logger.debug(f"Found chart from '{var_name}' variable")
                        break
            
            # Strategy 4: Find any Altair chart object in namespace
            if chart is None:
                chart_candidates = []
                for key, value in namespace.items():
                    if not key.startswith('_') and isinstance(value, alt.TopLevelMixin):
                        chart_candidates.append((key, value))
                
                if chart_candidates:
                    # If multiple candidates, prefer the last one or one with a title
                    titled_charts = [c for _, c in chart_candidates if hasattr(c, 'title') and c.title]
                    if titled_charts:
                        chart = titled_charts[0]
                    else:
                        chart = chart_candidates[-1][1]
                    logger.debug(f"Found chart from scanning namespace ({len(chart_candidates)} candidates)")
            
            if chart is None:
                error_msg = "No chart object found in executed code"
                logger.warning(f"Failed to extract chart from {chart_type}: {error_msg}")
                return None, error_msg
            
            # Render chart to PNG and encode to base64
            image_base64 = self._render_to_base64(chart)
            
            if image_base64 is None:
                error_msg = "Failed to render chart to PNG"
                logger.warning(f"Failed to render {chart_type}: {error_msg}")
                return None, error_msg
            
            logger.debug(f"Successfully executed and rendered {chart_type}")
            return image_base64, None
            
        except TimeoutError:
            error_msg = f"Execution timed out after {self.timeout} seconds"
            logger.warning(f"{chart_type}: {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            logger.warning(f"Failed to execute {chart_type}: {error_msg}")
            return None, error_msg
        finally:
            # Ensure alarm is cancelled
            try:
                signal.alarm(0)
            except AttributeError:
                pass
    
    def _prepare_namespace(self) -> dict:
        """
        Prepare isolated namespace with required dependencies.
        
        Returns:
            Dictionary with imported modules
        """
        namespace = {
            '__builtins__': __builtins__,
        }
        
        # Import common dependencies
        try:
            import altair as alt
            namespace['alt'] = alt
            namespace['altair'] = alt
        except ImportError:
            logger.error("altair not installed")
        
        try:
            import pandas as pd
            namespace['pd'] = pd
            namespace['pandas'] = pd
        except ImportError:
            logger.error("pandas not installed")
        
        try:
            import numpy as np
            namespace['np'] = np
            namespace['numpy'] = np
        except ImportError:
            logger.error("numpy not installed")
        
        try:
            from vega_datasets import data
            namespace['data'] = data
        except ImportError:
            logger.warning("vega_datasets not installed, some examples may fail")
        
        return namespace
    
    def _render_to_base64(self, chart) -> Optional[str]:
        """
        Render Altair chart to PNG and encode as base64.
        
        Args:
            chart: Altair chart object
            
        Returns:
            Base64-encoded PNG string, or None if rendering failed
        """
        try:
            # Use temporary file for rendering
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Render chart to PNG file
                chart.save(tmp_path, format='png', scale_factor=self.scale)
                
                # Read and encode to base64
                with open(tmp_path, 'rb') as f:
                    png_data = f.read()
                
                image_base64 = base64.b64encode(png_data).decode('utf-8')
                return image_base64
                
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to render chart: {type(e).__name__}: {str(e)}")
            return None


# Singleton instance
_executor: Optional[ChartExecutor] = None


def get_chart_executor(timeout: float = 5.0, scale: float = 1.5) -> ChartExecutor:
    """
    Get or create singleton chart executor.
    
    Args:
        timeout: Maximum execution time per chart (seconds)
        scale: Scale factor for rendered images
        
    Returns:
        ChartExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = ChartExecutor(timeout=timeout, scale=scale)
    return _executor
