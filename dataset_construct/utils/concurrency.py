"""
Adaptive Concurrency Controller for parallel candidate processing.
Dynamically adjusts concurrency based on API responses (rate limiting).
"""
import threading
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyState:
    """Concurrency control state"""
    current_workers: int
    consecutive_429_errors: int = 0
    consecutive_success: int = 0
    last_adjustment_time: float = 0.0
    total_429_errors: int = 0


class AdaptiveConcurrencyController:
    """
    Adaptive concurrency controller: dynamically adjusts concurrency based on API responses.
    
    Strategy:
    - Detect 429 error: immediately reduce concurrency (half)
    - Continuous success: gradually increase concurrency (add 1 every N successes)
    - Minimum/maximum limit
    """
    
    def __init__(
        self,
        initial_workers: int = 5,
        min_workers: int = 1,
        max_workers: int = 10,
        recovery_threshold: int = 20  # Try to increase concurrency after N consecutive successes
    ):
        self.initial_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.recovery_threshold = recovery_threshold
        
        self.state = ConcurrencyState(current_workers=initial_workers)
        self._lock = threading.Lock()
    
    def get_current_workers(self) -> int:
        """Get current recommended concurrency"""
        with self._lock:
            return self.state.current_workers
    
    def report_429_error(self):
        """Report encountering 429 error (API rate limiting)"""
        with self._lock:
            self.state.consecutive_429_errors += 1
            self.state.consecutive_success = 0
            self.state.total_429_errors += 1
            
            # Immediately reduce concurrency (half, but not below min_workers)
            old_workers = self.state.current_workers
            self.state.current_workers = max(
                self.min_workers,
                self.state.current_workers // 2
            )
            
            logger.warning(
                f"API rate limit detected (429). "
                f"Reducing concurrency: {old_workers} -> {self.state.current_workers}"
            )
            self.state.last_adjustment_time = time.time()
    
    def report_success(self):
        """Report successful completion of a task"""
        with self._lock:
            self.state.consecutive_success += 1
            self.state.consecutive_429_errors = 0
            
            # Continuous success reaches threshold, try to increase concurrency
            if self.state.consecutive_success >= self.recovery_threshold:
                if self.state.current_workers < self.max_workers:
                    old_workers = self.state.current_workers
                    self.state.current_workers = min(
                        self.max_workers,
                        self.state.current_workers + 1
                    )
                    logger.info(
                        f"Increasing concurrency after {self.recovery_threshold} successes: "
                        f"{old_workers} -> {self.state.current_workers}"
                    )
                    self.state.consecutive_success = 0
                    self.state.last_adjustment_time = time.time()
    
    def get_stats(self) -> dict:
        """Get statistics"""
        with self._lock:
            return {
                "current_workers": self.state.current_workers,
                "total_429_errors": self.state.total_429_errors,
                "consecutive_success": self.state.consecutive_success,
            }


# Global singleton
_concurrency_controller: Optional[AdaptiveConcurrencyController] = None


def get_concurrency_controller(reset: bool = False) -> AdaptiveConcurrencyController:
    """Get or create global concurrency controller"""
    global _concurrency_controller
    if _concurrency_controller is None or reset:
        from core.config import get_config
        config = get_config()
        _concurrency_controller = AdaptiveConcurrencyController(
            initial_workers=config.pipeline.initial_parallel_workers,
            min_workers=config.pipeline.min_parallel_workers,
            max_workers=config.pipeline.max_parallel_workers,
        )
    return _concurrency_controller
