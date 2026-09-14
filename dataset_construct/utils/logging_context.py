"""
Context management for candidate-level logging.

Provides thread-safe context tracking for parallel processing of candidates.
"""
import contextvars
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Context variables for tracking current processing context
_current_candidate_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('candidate_context', default=None)
)


def set_candidate_context(question_id: int, candidate_id: str = None, candidate_index: int = None):
    """
    Set the current candidate context for logging.
    
    Args:
        question_id: Source question ID
        candidate_id: Full candidate ID (e.g., "c1_multiple_series_line_chart"), None for instance-level
        candidate_index: Candidate index (0-based), None for instance-level
    """
    if candidate_id is None and candidate_index is None:
        # Instance-level context (for candidate generation phase)
        context = {
            'question_id': question_id,
            'candidate_id': None,
            'candidate_index': None,
            'log_prefix': f"q{question_id}_instance",
            'level': 'instance'
        }
    else:
        # Candidate-level context (for individual candidate processing)
        context = {
            'question_id': question_id,
            'candidate_id': candidate_id,
            'candidate_index': candidate_index,
            'log_prefix': f"q{question_id}_c{candidate_index}",
            'level': 'candidate'
        }
    _current_candidate_context.set(context)


def get_candidate_context() -> Optional[Dict[str, Any]]:
    """Get the current candidate context."""
    return _current_candidate_context.get()


def clear_candidate_context():
    """Clear the current candidate context."""
    _current_candidate_context.set(None)


@contextmanager
def candidate_logging_context(question_id: int, candidate_id: str = None, candidate_index: int = None):
    """
    Context manager for candidate-level logging.
    
    Usage:
        # Instance-level (for candidate generation)
        with candidate_logging_context(1471):
            # LLM calls logged to llm_io_..._q1471_instance.md
            generate_candidates(...)
        
        # Candidate-level (for processing specific candidate)
        with candidate_logging_context(1471, "c0_bar_chart", 0):
            # LLM calls logged to llm_io_..._q1471_c0.md
            process_candidate(...)
    """
    # Save previous context for restoration
    token = _current_candidate_context.set(None)  # Get reset token first
    set_candidate_context(question_id, candidate_id, candidate_index)
    try:
        yield
    finally:
        # Restore previous context using token
        _current_candidate_context.reset(token)


# Alias for clarity
def instance_logging_context(question_id: int):
    """
    Context manager for instance-level logging (candidate generation phase).
    
    Usage:
        with instance_logging_context(1471):
            # All LLM calls here will be logged to llm_io_..._q1471_instance.md
            candidates = generate_candidates(...)
    """
    return candidate_logging_context(question_id, candidate_id=None, candidate_index=None)


def get_llm_io_path() -> Optional[Path]:
    """
    Get the llm_io log path for current candidate context.
    
    Returns:
        Path to llm_io file in hierarchical directory structure, or None if no context is set
        
    Directory structure:
        output/logs/{run_id}/q{qid}/instance.md              (instance-level)
        output/logs/{run_id}/q{qid}/c{idx}/llm_io.md        (candidate-level)
        
    Note:
        Only returns path when inside a candidate_logging_context().
        Without context, returns None (no llm_io logging).
    """
    from core.config import get_config
    
    context = get_candidate_context()
    run_id = os.environ.get("VIS_INTERACT_RUN_ID")
    
    # Only log if we have both run_id and candidate context
    if not run_id or not context:
        return None
    
    config = get_config()
    base_log_dir = config.paths.logs_dir
    
    # Build hierarchical path: logs/{run_id}/q{qid}/...
    run_dir = base_log_dir / run_id
    question_id = context['question_id']
    instance_dir = run_dir / f"q{question_id}"
    
    if context['level'] == 'instance':
        # Instance-level: logs/{run_id}/q{qid}/instance.md
        instance_dir.mkdir(parents=True, exist_ok=True)
        return instance_dir / "instance.md"
    else:
        # Candidate-level: logs/{run_id}/q{qid}/c{idx}/llm_io.md
        candidate_idx = context['candidate_index']
        candidate_dir = instance_dir / f"c{candidate_idx}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        return candidate_dir / "llm_io.md"


def get_log_prefix() -> str:
    """
    Get the log prefix for current candidate context.
    
    Returns:
        Prefix string like "[q1471_c0]", "[q1471_instance]", or "" if no context
    """
    context = get_candidate_context()
    if context:
        return f"[{context['log_prefix']}] "
    return ""


# Custom log filter to add candidate context to all log records
class CandidateContextFilter(logging.Filter):
    """Add candidate context prefix to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add candidate prefix to message if context exists
        prefix = get_log_prefix()
        if prefix and not record.getMessage().startswith(prefix):
            record.msg = prefix + str(record.msg)
        return True


def install_candidate_filter_to_logger(logger: logging.Logger):
    """
    Install candidate context filter to a logger.
    
    This makes all log messages from this logger automatically include
    the candidate prefix when a context is active.
    """
    # Check if filter already installed
    for f in logger.filters:
        if isinstance(f, CandidateContextFilter):
            return
    
    logger.addFilter(CandidateContextFilter())
