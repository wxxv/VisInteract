"""
Dataset State Manager for smart resume functionality.

Manages dataset state by scanning samples_full.json and calculating remaining work.
Replaces the checkpoint-based approach with a simpler, file-based state management.

Features:
- Scan existing samples from samples_full.json
- Calculate remaining work for each instance
- Incremental sample appending with atomic operations
- Statistics management (dataset_statistic.json)
- Signal handling for graceful interrupts
"""
import json
import logging
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.models import DataInstance, VisSample

logger = logging.getLogger(__name__)


class DatasetStateManager:
    """
    Manages dataset state for smart resume functionality.
    
    Replaces checkpoint_manager with a simpler approach:
    - State is derived from samples_full.json
    - Statistics saved to dataset_statistic.json
    - No checkpoint files needed
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize dataset state manager.
        
        Args:
            output_dir: Directory containing samples_full.json and dataset_statistic.json
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_full_path = self.output_dir / "samples_full.json"
        self.statistic_path = self.output_dir / "dataset_statistic.json"
        
        self._interrupted = False
        self._interrupt_event = threading.Event()  # used for thread-to-thread interrupt signal passing
        self._lock = threading.Lock()
        self._original_sigint = None
        self._original_sigterm = None
    
    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt signal was received."""
        return self._interrupted or self._interrupt_event.is_set()
    
    def get_interrupt_event(self) -> threading.Event:
        """Get the interrupt event for checking in worker threads."""
        return self._interrupt_event
    
    def scan_existing_samples(self, samples_full_path: Optional[Path] = None) -> Dict[int, List[str]]:
        """
        Scan samples_full.json and return existing samples per instance.
        
        Args:
            samples_full_path: Path to samples_full.json (defaults to self.samples_full_path)
            
        Returns:
            {question_id: [sample_id1, sample_id2, ...]}
        """
        if samples_full_path is None:
            samples_full_path = self.samples_full_path
        
        if not samples_full_path.exists():
            logger.debug(f"No existing samples file found at {samples_full_path}")
            return {}
        
        try:
            with open(samples_full_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            
            if not isinstance(samples, list):
                logger.warning(f"Invalid samples_full.json format: expected list, got {type(samples)}")
                return {}
            
            # Group samples by source_question_id
            # Only count samples that passed validation
            existing = {}
            skipped_count = 0
            for sample in samples:
                question_id = sample.get("source_question_id")
                sample_id = sample.get("sample_id")
                validation_passed = sample.get("validation_passed", True)  # Default True for backward compatibility
                
                if question_id is None or sample_id is None:
                    logger.warning(f"Sample missing question_id or sample_id: {sample.get('sample_id', 'unknown')}")
                    continue
                
                # Skip samples that did not pass validation
                if not validation_passed:
                    logger.debug(f"Skipping sample {sample_id} (validation_passed=False)")
                    skipped_count += 1
                    continue
                
                if question_id not in existing:
                    existing[question_id] = []
                existing[question_id].append(sample_id)
            
            total_samples = sum(len(v) for v in existing.values())
            if skipped_count > 0:
                logger.info(f"Scanned {total_samples} valid samples from {len(existing)} instances (skipped {skipped_count} failed samples)")
            else:
                logger.info(f"Scanned {total_samples} existing samples from {len(existing)} instances")
            
            return existing
            
        except Exception as e:
            logger.error(f"Failed to scan existing samples: {e}")
            return {}
    
    def calculate_remaining_work(
        self,
        all_instances: List[DataInstance],
        num_candidates: int,
        existing_samples: Dict[int, List[str]]
    ) -> List[Tuple[DataInstance, int]]:
        """
        Calculate remaining work for each instance.
        
        Args:
            all_instances: All instances to process
            num_candidates: Target number of candidates per instance
            existing_samples: {question_id: [sample_id1, ...]} from scan_existing_samples()
            
        Returns:
            [(instance, remaining_count), ...]
            If instance already has 2 samples and num_candidates=5, remaining_count=3
        """
        work_plan = []
        
        for instance in all_instances:
            existing_count = len(existing_samples.get(instance.question_id, []))
            remaining = max(0, num_candidates - existing_count)
            
            if remaining > 0:
                work_plan.append((instance, remaining))
                logger.debug(
                    f"Instance {instance.question_id}: {existing_count} existing, "
                    f"{remaining} remaining (target: {num_candidates})"
                )
            else:
                logger.debug(
                    f"Instance {instance.question_id}: {existing_count} existing, "
                    f"skipping (target: {num_candidates})"
                )
        
        total_remaining = sum(count for _, count in work_plan)
        logger.info(
            f"Work plan: {len(work_plan)} instances need work, "
            f"{total_remaining} total candidates to generate"
        )
        
        return work_plan
    
    def append_sample_to_full(
        self,
        sample: VisSample,
        samples_full_path: Optional[Path] = None,
        samples_path: Optional[Path] = None
    ):
        """
        Append a successful sample to samples_full.json (and samples.json) immediately.
        Thread-safe.
        
        Args:
            sample: The VisSample to append
            samples_full_path: Path to samples_full.json
            samples_path: Optional path to samples.json (if provided, will also be updated)
        """
        if samples_full_path is None:
            samples_full_path = self.samples_full_path
        
        # Safety check: only save samples that passed validation
        if not sample.validation_passed:
            logger.warning(f"Attempted to save sample {sample.sample_id} with validation_passed=False. Skipping.")
            return
        
        with self._lock:
            try:
                # Helper function to append to a file
                def _append_to_file(path: Path, format: str = "full"):
                    existing = []
                    if path.exists():
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                        except Exception:
                            logger.warning(f"Invalid format in {path.name}, resetting")
                    
                    if not isinstance(existing, list):
                        existing = []
                        
                    # Use specified format for conversion
                    sample_dict = sample.to_dict(format=format)
                    existing.append(sample_dict)
                    
                    # Atomic write
                    temp_path = path.parent / (path.name + ".tmp")
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(existing, f, indent=2, ensure_ascii=False)
                    temp_path.replace(path)
                    
                # 1. Update samples_full.json with full format
                _append_to_file(samples_full_path, format="full")
                logger.debug(f"Appended sample {sample.sample_id} to samples_full.json (full format)")
                
                # 2. Update samples.json with compact format if provided
                if samples_path:
                    _append_to_file(samples_path, format="compact")
                    logger.debug(f"Appended sample {sample.sample_id} to {samples_path.name} (compact format)")
                
            except Exception as e:
                logger.error(f"Failed to append sample: {e}")
                raise
    
    def save_statistics(
        self,
        statistics: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """
        Save complete statistics to dataset_statistic.json.
        
        Args:
            statistics: Complete statistics dictionary (includes timestamp, summary, distribution, diversity, instance_details)
            output_path: Path to save (defaults to self.statistic_path)
        """
        if output_path is None:
            output_path = self.statistic_path
        
        with self._lock:
            try:
                # Atomic write
                temp_path = output_path.parent / (output_path.name + ".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(statistics, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(output_path)
                
                logger.debug(f"Saved statistics to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save statistics: {e}")
                raise
    
    def load_statistics(self, statistic_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Load statistics from dataset_statistic.json.
        
        Args:
            statistic_path: Path to load (defaults to self.statistic_path)
            
        Returns:
            Statistics dictionary or None if file doesn't exist
        """
        if statistic_path is None:
            statistic_path = self.statistic_path
        
        if not statistic_path.exists():
            return None
        
        try:
            with open(statistic_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load statistics: {e}")
            return None
    
    def setup_signal_handlers(self):
        """
        Set up signal handlers for IMMEDIATE interrupt handling.
        Captures SIGINT (Ctrl+C) and SIGTERM, and FORCE EXITS the process.
        """
        import os
        
        def signal_handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            # Use print to ensure output before logging system is closed
            print(f"\n\n🛑 Received {sig_name}. Force exiting immediately to stop API calls...")
            
            # Force exit process, without Python-level cleanup (like finally blocks)
            # This is the only way to immediately cut off threads blocked in the C layer socket (OpenAI API)
            os._exit(0)
        
        # Save original handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        # Install new handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.debug("Signal handlers installed for graceful shutdown")
    
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
