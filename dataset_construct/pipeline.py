"""
Pipeline for Vis-Interact Dataset Construction.

Main orchestration for the new architecture:
1. Load DataInstance from BIRD Mini-Dev
2. Semantic Analysis
3. Generate N diverse Key Feature Candidates
4. For each candidate:
   - Transform SQL for visualization
   - Transform NL with ambiguity injection
   - Generate Altair code
   - Validate

One BIRD instance produces N visualization samples.

Features:
- Checkpoint support for resume after interruption
- Auto-save progress after each instance
- Graceful interrupt handling (Ctrl+C)
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd
import numpy as np


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas/numpy types."""
    
    def default(self, obj):
        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle numpy datetime64
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).isoformat()
        # Handle datetime
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle pandas NA/NaT
        if pd.isna(obj):
            return None
        return super().default(obj)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from core.models import (
    DataInstance, VisSample, FeatureCandidate, SemanticContext,
    DatabaseInfo, ValidationResult, IterationLog, IterationFeedback,
    AmbiguityProfile, Feature, RequiredKeyFeature,
)
from core.config import get_config, init_config
from utils.state_manager import DatasetStateManager
from generation.transform import get_transform_processor, TransformResult
from utils.diversity import DiversityTracker, DiversityConstraints, get_diversity_tracker
from generation.ambiguity import AmbiguityInjector, AmbiguityDiversityConstraints, get_ambiguity_injector
from utils.logging_context import candidate_logging_context, instance_logging_context

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution for a single sample."""
    success: bool = False
    sample: Optional[VisSample] = None
    error: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    steps_completed: List[str] = field(default_factory=list)
    iteration_log: Optional[IterationLog] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "sample": self.sample.to_dict() if self.sample else None,
            "error": self.error,
            "validation_result": {
                "passed": self.validation_result.passed,
                "failure_codes": self.validation_result.failure_codes
            } if self.validation_result else None,
            "steps_completed": self.steps_completed,
            "iteration_log": self.iteration_log.to_dict() if self.iteration_log else None
        }


@dataclass
class InstanceResult:
    """Result of processing one DataInstance (produces multiple samples)."""
    instance: DataInstance
    samples: List[VisSample] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    candidates_attempted: int = 0
    candidates_succeeded: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.instance.question_id,
            "db_id": self.instance.db_id,
            "samples_generated": len(self.samples),
            "candidates_attempted": self.candidates_attempted,
            "candidates_succeeded": self.candidates_succeeded,
            "errors": self.errors
        }


class Pipeline:
    """
    Main pipeline for Vis-Interact dataset construction.
    
    New architecture: One DataInstance → N VisSamples
    
    Steps:
    1. Load DataInstance from BIRD Mini-Dev
    2. Semantic Analysis (build SemanticContext)
    3. Generate N diverse Key Feature Candidates
    4. For each candidate:
       a. Transform SQL for visualization
       b. Execute SQL on SQLite
       c. Transform NL with ambiguity injection
       d. Generate Altair code
       e. Validate
    5. Output N VisSamples
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Core components (lazy initialization)
        self._preprocessor = None
        self._schema_processor = None
        self._sql_processor = None
        self._sqlite_client = None
        self._key_feature_gen = None
        self._required_key_feature_gen = None
        self._spec_generator = None
        self._validator = None
        self._transform_processor = None
        self._diversity_tracker = None
        self._ambiguity_injector = None
    
    # Lazy property accessors
    @property
    def preprocessor(self):
        if self._preprocessor is None:
            from preprocessing.loader import get_preprocessor
            self._preprocessor = get_preprocessor()
        return self._preprocessor
    
    @property
    def schema_processor(self):
        if self._schema_processor is None:
            from preprocessing.schema import get_schema_processor
            self._schema_processor = get_schema_processor()
        return self._schema_processor
    
    @property
    def sql_processor(self):
        if self._sql_processor is None:
            from preprocessing.sql_analyzer import get_sql_processor
            self._sql_processor = get_sql_processor()
        return self._sql_processor
    
    @property
    def sqlite_client(self):
        if self._sqlite_client is None:
            from execution.sqlite_client import get_sqlite_client
            self._sqlite_client = get_sqlite_client()
        return self._sqlite_client
    
    @property
    def key_feature_gen(self):
        if self._key_feature_gen is None:
            from generation.candidates import get_key_feature_generator
            self._key_feature_gen = get_key_feature_generator()
        return self._key_feature_gen
    
    @property
    def required_key_feature_gen(self):
        """Get or create the required key feature generator for NL-based features."""
        if self._required_key_feature_gen is None:
            from generation.features import get_required_key_feature_generator
            self._required_key_feature_gen = get_required_key_feature_generator()
        return self._required_key_feature_gen
    
    @property
    def spec_generator(self):
        if self._spec_generator is None:
            from generation.spec import get_spec_generator
            self._spec_generator = get_spec_generator()
        return self._spec_generator
    
    @property
    def validator(self):
        if self._validator is None:
            from validation.validator import get_validator
            self._validator = get_validator()
        return self._validator
    
    @property
    def transform_processor(self):
        if self._transform_processor is None:
            from generation.transform import get_transform_processor
            self._transform_processor = get_transform_processor()
            # Set SQL executor for validation
            self._transform_processor.set_sql_executor_with_error(self.sqlite_client.execute_with_error)
        return self._transform_processor
    
    @property
    def diversity_tracker(self) -> DiversityTracker:
        """Get or create the diversity tracker for global sample diversity control."""
        if self._diversity_tracker is None:
            self._diversity_tracker = get_diversity_tracker()
        return self._diversity_tracker
    
    @property
    def ambiguity_injector(self) -> AmbiguityInjector:
        """Get or create the ambiguity injector for post-Altair ambiguity injection."""
        if self._ambiguity_injector is None:
            self._ambiguity_injector = get_ambiguity_injector()
        return self._ambiguity_injector
    
    # =========================================================================
    # Main Entry Point: Process Instance
    # =========================================================================
    
    def process_single_candidate(
        self,
        instance: DataInstance,
        semantic_ctx: Optional[SemanticContext] = None,
        candidate_index: int = 0
    ) -> Optional[VisSample]:
        """
        Process a single candidate for an instance.
        Used by the new smart resume workflow to generate one candidate at a time.
        
        Args:
            instance: DataInstance to process
            semantic_ctx: Pre-computed semantic context (if None, will compute)
            candidate_index: Index of the candidate (for sample_id generation)
            
        Returns:
            VisSample if successful, None otherwise
        """
        debug_mode = self.config.pipeline.debug_mode
        
        try:
            # Step 1: Semantic Analysis (if not provided)
            if semantic_ctx is None:
                logger.info(f"[{instance.question_id}] Step 1: Semantic analysis...")
                semantic_ctx = self.preprocessor.analyze_instance(instance)
                
                if semantic_ctx.schema_info is None:
                    logger.warning(f"[{instance.question_id}] Failed to build schema")
                    return None
            
            # Step 2: Generate a single diverse candidate
            with instance_logging_context(instance.question_id):
                diversity_constraints = self.diversity_tracker.get_diversity_constraints(
                    all_chart_types=self.key_feature_gen.ALL_CHART_TYPES
                )
                
                # Generate a single candidate
                candidates = self.key_feature_gen.generate_candidates(
                    semantic_ctx,
                    num_initial=1,
                    num_final=1,
                    diversity_constraints=diversity_constraints
                )
            
            if not candidates:
                logger.warning(f"[{instance.question_id}] No candidates generated")
                return None
            
            candidate = candidates[0]
            logger.info(f"[{instance.question_id}] Processing candidate: {candidate.chart_type}")
            
            # Step 3-5: Process the candidate
            with candidate_logging_context(
                question_id=instance.question_id,
                candidate_id=candidate.candidate_id,
                candidate_index=candidate_index
            ):
                sample = self._process_candidate(
                    instance=instance,
                    candidate=candidate,
                    semantic_ctx=semantic_ctx,
                    candidate_index=candidate_index
                )
                
                if sample:
                    logger.info(f"  ✓ Candidate succeeded")
                    return sample
                else:
                    logger.warning(f"  ✗ Candidate failed")
                    return None
                    
        except Exception as e:
            if debug_mode:
                raise
            logger.error(f"[{instance.question_id}] Error processing candidate: {e}")
            return None
    
    def process_instance_parallel(
        self,
        instance: DataInstance,
        num_candidates: int = 5,
        state_mgr = None,
        samples_full_path: Optional[Path] = None,
        samples_path: Optional[Path] = None,
        start_index: int = 0
    ) -> InstanceResult:
        """
        Parallel processing of multiple candidates for an instance (adaptive concurrency).
        
        Flow:
        1. Serial: Semantic Analysis (only once)
        2. Serial: Generate N diverse candidates (only once)
        3. Parallel: Call _process_candidate_parallel() for each candidate
        4. Collect results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.concurrency import get_concurrency_controller
        from utils.llm_client import RateLimitError
        
        result = InstanceResult(instance=instance)
        
        # Step 1: Semantic Analysis (serial, only once)
        with instance_logging_context(instance.question_id):
            semantic_ctx = self.preprocessor.analyze_instance(instance)
            
            if semantic_ctx.schema_info is None:
                logger.warning(f"[{instance.question_id}] Failed to build schema")
                return result
            
            # Step 2: Generate N diverse candidates (serial, only once)
            # 使用预占机制获取多样性约束
            reservation_id, diversity_constraints = self.diversity_tracker.reserve_recommendations(
                all_chart_types=self.key_feature_gen.ALL_CHART_TYPES
            )
            
            logger.info(f"[{instance.question_id}] Generating {num_candidates} diverse candidates...")
            candidates = self.key_feature_gen.generate_candidates(
                semantic_ctx,
                num_initial=num_candidates,  # 直接生成目标数量
                num_final=num_candidates,
                diversity_constraints=diversity_constraints
            )
            
            if not candidates:
                logger.warning(f"[{instance.question_id}] No candidates generated")
                self.diversity_tracker.commit_selection(reservation_id, None)
                return result
            
            logger.info(f"[{instance.question_id}] Generated {len(candidates)} candidates: {[c.chart_type for c in candidates]}")
        
        # Get concurrency controller
        concurrency_ctrl = get_concurrency_controller()
        
        # Use the incoming state_mgr or create a new one (for independent calling)
        if state_mgr is None:
            from utils.state_manager import DatasetStateManager
            state_mgr = DatasetStateManager(self.config.paths.output_dir)
        
        # Step 3: Parallel processing of all candidates (adaptive concurrency)
        max_workers = min(len(candidates), concurrency_ctrl.get_current_workers())
        
        logger.info(f"[{instance.question_id}] Starting parallel processing with {max_workers} workers")
        
        # Assign sample_id to each candidate
        for i, candidate in enumerate(candidates):
            idx = start_index + i
            candidate.candidate_id = f"q{instance.question_id}_c{idx}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all candidate tasks
            futures = {}
            for i, candidate in enumerate(candidates):
                idx = start_index + i
                future = executor.submit(
                    self._process_candidate_parallel,
                    instance, candidate, semantic_ctx, idx
                )
                futures[future] = (idx, candidate)
            
            # Collect results (using polling to check for interruption)
            pending_futures = set(futures.keys())
            while pending_futures:
                # Check interruption flag
                if state_mgr.is_interrupted:
                    logger.warning(f"[{instance.question_id}] Interrupted, cancelling remaining tasks...")
                    # Cancel all unfinished tasks
                    for future in pending_futures:
                        future.cancel()
                    break
                
                # Non-blocking check for completed futures (timeout=0.5 seconds)
                done_futures = set()
                for future in pending_futures:
                    if future.done():
                        done_futures.add(future)
                
                # Process completed futures
                for future in done_futures:
                    pending_futures.remove(future)
                    candidate_index, candidate = futures[future]
                    
                    try:
                        sample = future.result()
                        result.candidates_attempted += 1
                        
                        if sample:
                            result.samples.append(sample)
                            result.candidates_succeeded += 1
                            concurrency_ctrl.report_success()  # Report success
                            
                            # Only record successful samples to diversity tracker
                            self.diversity_tracker.record_sample(sample)
                            
                            logger.info(f"[q{instance.question_id}] Candidate {candidate_index} ({candidate.chart_type}) succeeded")
                            
                            # Immediately save (incremental save in parallel mode)
                            if samples_full_path and state_mgr:
                                state_mgr.append_sample_to_full(sample, samples_full_path, samples_path)
                        else:
                            logger.warning(f"[q{instance.question_id}] Candidate {candidate_index} ({candidate.chart_type}) failed")
                            
                    except KeyboardInterrupt:
                        # User interrupt - immediately propagate, exit loop
                        logger.warning(f"[q{instance.question_id}] Candidate {candidate_index} interrupted by user")
                        # Set interruption flag (if not already set)
                        if state_mgr:
                            state_mgr._interrupt_event.set()
                        raise
                        
                    except RateLimitError as e:
                        # 429 error: report to concurrency controller
                        concurrency_ctrl.report_429_error()
                        result.candidates_attempted += 1
                        result.errors.append(f"Candidate {candidate_index}: Rate limit - {str(e)}")
                        logger.error(f"[q{instance.question_id}] Candidate {candidate_index} rate limited")
                        
                    except Exception as e:
                        result.candidates_attempted += 1
                        result.errors.append(f"Candidate {candidate_index}: {str(e)}")
                        logger.error(f"[q{instance.question_id}] Candidate {candidate_index} error: {e}")
                
                # If there are pending futures, check again after a short sleep
                if pending_futures:
                    import time
                    time.sleep(0.5)  # Check every 0.5 seconds
        
        # Clean up reservation records (after all candidates are processed)
        # Roll back the reservation count, because record_sample has already re-counted the actual generated samples
        # So the final count is correct: reservation+1 → record_sample+1 → rollback-1 = final+1
        self.diversity_tracker.commit_selection(reservation_id, None, skip_rollback=False)
        
        # Record concurrency statistics
        stats = concurrency_ctrl.get_stats()
        logger.debug(f"[{instance.question_id}] Concurrency stats: {stats}")
        
        return result
    
    def _calculate_start_index(
        self,
        instance: DataInstance,
        existing_samples: Dict[int, List[str]]
    ) -> int:
        """
        Calculate the starting candidate index for an instance to avoid overwriting existing samples.
        
        Args:
            instance: DataInstance to process
            existing_samples: {question_id: [sample_id1, ...]} from scan_existing_samples()
        
        Returns:
            start_index: Starting index (e.g., if c0,c1,c2 exist, returns 3)
        """
        existing_sample_ids = existing_samples.get(instance.question_id, [])
        if not existing_sample_ids:
            return 0
        
        # Extract all indices from sample_ids
        indices = []
        for sid in existing_sample_ids:
            # sid format: q{qid}_c{idx}
            if '_c' in sid:
                try:
                    idx = int(sid.split('_c')[-1])
                    indices.append(idx)
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse index from sample_id: {sid}")
        
        # Return max index + 1, or 0 if no valid indices
        return max(indices) + 1 if indices else 0
    
    def _process_instances_serial(
        self,
        work_plan: List[tuple],
        num_candidates: int,
        state_mgr,
        samples_full_path: Path,
        samples_path: Path,
        statistic_path: Path,
        existing_samples: Dict[int, List[str]],
        parallel: bool,
        show_progress: bool
    ) -> List[InstanceResult]:
        """
        Process instances serially (original behavior).
        Each instance is processed one after another, but candidates within
        an instance can be processed in parallel if parallel=True.
        
        Args:
            work_plan: List of (instance, remaining_count) tuples
            num_candidates: Target number of candidates per instance (for logging)
            state_mgr: DatasetStateManager for interrupt handling
            samples_full_path: Path to samples_full.json
            samples_path: Path to samples.json
            statistic_path: Path to dataset_statistic.json
            existing_samples: Cached existing samples {question_id: [sample_ids]}
            parallel: Whether to use parallel processing for candidates
            show_progress: Whether to show progress bar
            
        Returns:
            List of InstanceResult
        """
        results: List[InstanceResult] = []
        instance_results_map: Dict[int, InstanceResult] = {}
        semantic_ctx_cache: Dict[int, SemanticContext] = {}
        
        try:
            if show_progress and tqdm is not None:
                work_iter = tqdm(work_plan, desc="Processing instances", unit="instance")
            else:
                work_iter = work_plan
            
            for instance, remaining_count in work_iter:
                if state_mgr.is_interrupted:
                    logger.warning("⚠️  Interrupted. Stopping...")
                    break
                
                logger.info(f"Processing {instance.question_id}: generating {remaining_count} candidates {'(parallel)' if parallel else '(serial)'}")
                
                # Calculate start_index using helper method
                start_index = self._calculate_start_index(instance, existing_samples)
                
                if parallel:
                    # Parallel processing of all candidates (pass state_mgr to check for interruption and perform incremental save)
                    instance_result = self.process_instance_parallel(
                        instance, remaining_count, state_mgr=state_mgr,
                        samples_full_path=samples_full_path,
                        samples_path=samples_path,
                        start_index=start_index
                    )
                    
                    # Update statistics (if samples have increased)
                    if instance_result.samples:
                        stats = self._build_statistics_incremental()
                        state_mgr.save_statistics(stats, statistic_path)
                    
                    results.append(instance_result)
                    
                else:
                    # Serial processing (original logic)
                    # Get or compute semantic context (cache for efficiency)
                    if instance.question_id not in semantic_ctx_cache:
                        semantic_ctx_cache[instance.question_id] = self.preprocessor.analyze_instance(instance)
                    
                    semantic_ctx = semantic_ctx_cache[instance.question_id]
                    if semantic_ctx.schema_info is None:
                        logger.warning(f"[{instance.question_id}] Failed to build schema, skipping")
                        continue
                    
                    # Initialize result for this instance
                    if instance.question_id not in instance_results_map:
                        instance_results_map[instance.question_id] = InstanceResult(instance=instance)
                    result = instance_results_map[instance.question_id]
                    
                    # Generate remaining candidates serially
                    for i in range(remaining_count):
                        if state_mgr.is_interrupted:
                            break
                        
                        try:
                            # Pass correct candidate_index: start_index + i
                            sample = self.process_single_candidate(
                                instance, 
                                semantic_ctx,
                                candidate_index=start_index + i
                            )
                            
                            if sample:  # Only save successful samples
                                # Immediately append to samples_full.json and samples.json
                                state_mgr.append_sample_to_full(sample, samples_full_path, samples_path)
                                
                                # Update diversity tracker
                                self.diversity_tracker.record_sample(sample)
                                
                                # Update result
                                result.samples.append(sample)
                                result.candidates_succeeded += 1
                                
                                # Update statistics incrementally
                                stats = self._build_statistics_incremental()
                                state_mgr.save_statistics(stats, statistic_path)
                                
                                logger.info(f"  ✓ Candidate {i+1}/{remaining_count} saved")
                            else:
                                logger.warning(f"  ✗ Candidate {i+1}/{remaining_count} failed")
                            
                            result.candidates_attempted += 1
                            
                        except Exception as e:
                            debug_mode = self.config.pipeline.debug_mode
                            if debug_mode:
                                raise  # Re-raise in debug mode
                            logger.error(f"  Error in candidate {i+1}: {e}")
                            result.candidates_attempted += 1
                            result.errors.append(f"Candidate {i+1} error: {str(e)}")
                            continue
                    
                    # Add result to list if not already there
                    if result not in results:
                        results.append(result)
                
                if show_progress and tqdm is not None:
                    total_samples = sum(len(r.samples) for r in results)
                    total_attempted = sum(r.candidates_attempted for r in results)
                    work_iter.set_postfix({
                        "samples": total_samples,
                        "success_rate": f"{total_samples / max(total_attempted, 1) * 100:.1f}%"
                    })
        
        except KeyboardInterrupt:
            # User interrupt - save completed work
            logger.warning("\n⚠️  Process interrupted by user. Saving completed work...")
            # Completed samples have been saved, statistics have also been incrementally updated
            # No additional processing needed, just exit
        
        return results
    
    def _process_single_instance_wrapper(
        self,
        instance: DataInstance,
        remaining_count: int,
        state_mgr,
        samples_full_path: Path,
        samples_path: Path,
        start_index: int,
        statistic_path: Path
    ) -> InstanceResult:
        """
        Wrapper for processing a single instance in a thread.
        Sets up logging context and calls process_instance_parallel.
        
        Args:
            instance: DataInstance to process
            remaining_count: Number of candidates to generate
            state_mgr: DatasetStateManager for interrupt handling
            samples_full_path: Path to samples_full.json
            samples_path: Path to samples.json
            start_index: Starting index for candidate numbering
            statistic_path: Path to dataset_statistic.json
            
        Returns:
            InstanceResult
        """
        try:
            # Set thread-level logging context
            with instance_logging_context(instance.question_id):
                result = self.process_instance_parallel(
                    instance=instance,
                    num_candidates=remaining_count,
                    state_mgr=state_mgr,
                    samples_full_path=samples_full_path,
                    samples_path=samples_path,
                    start_index=start_index
                )
                
                # Update statistics if samples were generated
                if result.samples:
                    stats = self._build_statistics_incremental()
                    state_mgr.save_statistics(stats, statistic_path)
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing instance {instance.question_id}: {e}")
            # Return a failed InstanceResult
            result = InstanceResult(instance=instance)
            result.errors.append(str(e))
            return result
    
    def _process_instances_parallel(
        self,
        work_plan: List[tuple],
        num_candidates: int,
        state_mgr,
        samples_full_path: Path,
        samples_path: Path,
        statistic_path: Path,
        existing_samples: Dict[int, List[str]],
        show_progress: bool
    ) -> List[InstanceResult]:
        """
        Process multiple instances in parallel (instance-level parallelism).
        Each instance internally uses process_instance_parallel for candidate-level parallelism.
        
        Args:
            work_plan: List of (instance, remaining_count) tuples
            num_candidates: Target number of candidates per instance (for logging)
            state_mgr: DatasetStateManager for interrupt handling
            samples_full_path: Path to samples_full.json
            samples_path: Path to samples.json
            statistic_path: Path to dataset_statistic.json
            existing_samples: Cached existing samples {question_id: [sample_ids]}
            show_progress: Whether to show progress bar
            
        Returns:
            List of InstanceResult
        """
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        results = []
        instance_workers = self.config.pipeline.instance_parallel_workers
        
        logger.info(f"Starting instance-level parallel processing with {instance_workers} workers")
        
        # Create progress bar if requested
        progress_bar = None
        if show_progress and tqdm is not None:
            progress_bar = tqdm(total=len(work_plan), desc="Processing instances", unit="instance")
        
        with ThreadPoolExecutor(max_workers=instance_workers) as executor:
            # Submit all instance tasks
            futures = {}
            for instance, remaining_count in work_plan:
                # Calculate start_index using cached existing_samples
                start_index = self._calculate_start_index(instance, existing_samples)
                
                future = executor.submit(
                    self._process_single_instance_wrapper,
                    instance, remaining_count, state_mgr,
                    samples_full_path, samples_path, start_index, statistic_path
                )
                futures[future] = instance
            
            # Poll for completion (supports interrupt detection)
            pending_futures = set(futures.keys())
            completed_count = 0
            total_count = len(futures)
            
            while pending_futures:
                # Check interrupt flag
                if state_mgr.is_interrupted:
                    logger.warning("⚠️  Interrupted at instance level, cancelling remaining tasks...")
                    for f in pending_futures:
                        f.cancel()
                    break
                
                # Non-blocking check for completed futures
                done_futures = {f for f in pending_futures if f.done()}
                
                for future in done_futures:
                    pending_futures.remove(future)
                    instance = futures[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress bar
                        if progress_bar is not None:
                            total_samples = sum(len(r.samples) for r in results)
                            total_attempted = sum(r.candidates_attempted for r in results)
                            progress_bar.update(1)
                            progress_bar.set_postfix({
                                'samples': total_samples,
                                'attempted': total_attempted,
                                'success': f"{total_samples}/{total_attempted}" if total_attempted > 0 else "0/0"
                            })
                        
                        success_rate = f"{result.candidates_succeeded}/{result.candidates_attempted}"
                        logger.info(f"[{completed_count}/{total_count}] Instance {instance.question_id} completed: {success_rate} candidates succeeded")
                        
                    except KeyboardInterrupt:
                        # User interrupt - propagate
                        logger.warning(f"Instance {instance.question_id} interrupted by user")
                        state_mgr._interrupt_event.set()
                        raise
                        
                    except Exception as e:
                        # Log error but continue processing other instances
                        logger.error(f"Instance {instance.question_id} error: {e}")
                        # Create a failed result
                        failed_result = InstanceResult(instance=instance)
                        failed_result.errors.append(str(e))
                        results.append(failed_result)
                
                # Sleep briefly if there are still pending futures
                if pending_futures:
                    time.sleep(0.5)
        
        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()
        
        logger.info(f"Instance-level parallel processing completed: {len(results)} instances processed")
        return results
    
    def _process_candidate_parallel(
        self,
        instance: DataInstance,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        candidate_index: int
    ) -> Optional[VisSample]:
        """
        Parallel processing of a single generated candidate (thread-safe).
        
        Note: candidate has already been generated at the instance level, here only responsible for processing.
        
        Flow:
        1. Process: Process candidate (transform + spec + ambiguity)
        2. Return result
        """
        try:
            # Set thread-level logging context
            with candidate_logging_context(
                question_id=instance.question_id,
                candidate_id=f"c{candidate_index}",
                candidate_index=candidate_index
            ):
                # Process candidate (original _process_candidate logic)
                sample = self._process_candidate(
                    instance, candidate, semantic_ctx, candidate_index
                )
                
                return sample
                
        except Exception as e:
            logger.error(f"Error in candidate {candidate_index}: {e}")
            raise  # Re-raise to be caught by ThreadPoolExecutor
    
    def _process_candidate(
        self,
        instance: DataInstance,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        candidate_index: int
    ) -> Optional[VisSample]:
        """
        Process a single candidate to generate one VisSample.
        
        Flow:
        1. Transform SQL and NL (clear version, no ambiguity)
        2. Execute SQL
        3. Generate Altair code using clear NL
        4. AFTER Altair success, inject ambiguity
        5. Build sample with both clear and ambiguous NL
        """
        # Build sample_id early for image naming
        sample_id = f"q{instance.question_id}_c{candidate_index}"
        
        # Force sync candidate_id with sample_id to ensure consistency
        # (candidate_id may have been set incorrectly by generate_candidates)
        candidate.candidate_id = sample_id
        
        # Step 3-4: Transform - Generate complete data processing code + NL
        # Clear NL is used for Altair generation
        # Get database paths: absolute for execution, relative for prompts
        db_path_absolute = self.sqlite_client.get_db_path(instance.db_id)
        db_path_for_llm = self.sqlite_client.get_db_path_for_llm(instance.db_id)
        
        transform_result = self.transform_processor.transform(
            candidate=candidate,
            semantic_ctx=semantic_ctx,
            schema_info=semantic_ctx.schema_info,
            min_rows=self.config.pipeline.min_data_rows,
            max_rows=self.config.pipeline.max_data_rows,
            db_path=db_path_absolute,  # Absolute path for code execution
            validate_execution=True,
            max_retries=self.config.feedback.max_sql_retries
        )
        
        if not transform_result.success:
            logger.warning(f"    Transform failed: {transform_result.error or 'unknown error'}")
            return None
        
        if not transform_result.execution_validated:
            logger.warning("    Data processing validation did not pass, skipping candidate")
            return None
        
        if transform_result.data_quality_passed is False:
            logger.warning("    Data quality validation failed, skipping candidate")
            return None
        
        # Extract results from transform
        data_processing_code = transform_result.data_processing_code
        vis_question_clear = transform_result.vis_question
        data = transform_result.processed_df
        data_quality_report = transform_result.data_quality_report
        
        if data is None or data.empty:
            logger.warning(f"    Transform returned no data")
            return None
        
        # Step 5: Generate Altair code with validation
        # Uses the CLEAR question for accurate code generation
        altair_code, vega_spec, image_base64, image_path, vlm_report, code_success, vlm_validation_passed, code_error = self.spec_generator.generate_altair_code_with_validation(
            df=data,
            candidate=candidate,
            data_processing_code=data_processing_code,
            schema_info=semantic_ctx.schema_info,
            sql_summary=semantic_ctx.sql_summary,
            user_question=vis_question_clear,  # Use clear question
            evidence=instance.evidence or "",
            max_retries=self.config.feedback.max_code_retries,
            sample_id=sample_id,
            db_path=db_path_for_llm  # Relative path for prompts
        )
        
        if not code_success:
            logger.warning(f"    Altair code generation/validation failed: {code_error or 'unknown error'}")
            return None
        
        if vega_spec is None:
            logger.warning(f"    Altair code generation failed after retries")
            return None
        
        # Check VLM validation result (if enabled)
        # If VLM validation fails after retries, reject the candidate to avoid wasting resources
        if self.config.validation.enable_vlm_chart_check:
            if vlm_validation_passed is False:
                # VLM validation failed after multiple retries - reject candidate
                logger.warning("    VLM validation failed after retries, skipping candidate")
                return None
            elif vlm_validation_passed is None:
                logger.info("    VLM validation skipped (parsing error)")
            else:
                logger.info("    VLM validation passed")
        else:
            logger.debug("    VLM validation disabled in config")
        
        # Step 5.5: Extract actual features from the generated vega_spec
        # This ensures features reflect the actual visualization, not just the intended design
        extracted_features = self.spec_generator.extract_features_from_spec(vega_spec)
        logger.debug(f"    Extracted {len(extracted_features)} features from spec")
        
        # Step 5.6: Generate RequiredKeyFeature (NL-based) for metric evaluation
        # Uses LLM to analyze vega_spec and generate user-facing requirements
        required_key_features = self.required_key_feature_gen.generate(
            vega_spec=vega_spec,
            candidate=candidate,
            semantic_ctx=semantic_ctx
        )
        logger.debug(f"    Generated {len(required_key_features)} required key features")
        
        # Step 6: Inject ambiguity AFTER Altair success
        # Get ambiguity constraints from diversity tracker
        ambiguity_constraints_dict = self.diversity_tracker.get_ambiguity_constraints()
        ambiguity_constraints = AmbiguityDiversityConstraints(
            preferred_types=ambiguity_constraints_dict.get("preferred_types", []),
            avoid_types=ambiguity_constraints_dict.get("avoid_types", []),
            preferred_templates=ambiguity_constraints_dict.get("preferred_templates", []),
            avoid_templates=ambiguity_constraints_dict.get("avoid_templates", []),
            strength=ambiguity_constraints_dict.get("strength", 0.5)
        )
        
        # Pass required_key_features to guide ambiguity injection
        vis_question, ambiguity_profile = self.ambiguity_injector.inject(
            clear_question=vis_question_clear,
            candidate=candidate,
            vega_spec=vega_spec,
            semantic_ctx=semantic_ctx,
            diversity_constraints=ambiguity_constraints,
            required_key_features=required_key_features
        )
        
        # Build sample with both clear and ambiguous questions
        sample = VisSample(
            sample_id=sample_id,
            source_question_id=instance.question_id,
            candidate_id=candidate.candidate_id,
            source_instance=instance,
            candidate=candidate,
            vis_sql="",  # Deprecated, kept for backward compatibility
            data_processing_code=data_processing_code,
            vis_question=vis_question,  # Ambiguous version (for benchmark)
            vis_question_clear=vis_question_clear,  # Clear version
            original_question=instance.question,
            ambiguity_profile=ambiguity_profile,
            data=data.to_dict(orient="records"),
            altair_code=altair_code,  # Complete code (data processing + Altair)
            vega_lite_spec=vega_spec,
            features=extracted_features,  # Spec validation features (path-op-value)
            required_key_features=required_key_features,  # NL-based features for metric
            chart_type=candidate.chart_type,
            chart_category=candidate.chart_category,
            vis_intent_type=candidate.vis_intent_type.value,
            db_id=instance.db_id,
            difficulty=instance.difficulty,
            validation_passed=True,  # Will be updated by validation
            # Validation report fields
            data_quality_report=data_quality_report,
            vlm_validation_report=vlm_report,
            chart_image_path=image_path
        )
        
        # Validate
        validation_passed = self._validate_sample(sample)
        sample.validation_passed = validation_passed
        
        if not validation_passed:
            logger.warning(f"    Sample validation failed")
            return None
        
        return sample
    
    def _validate_sample(self, sample: VisSample) -> bool:
        """Validate a sample for correctness."""
        try:
            # Basic validations
            if not sample.data or len(sample.data) == 0:
                return False
            
            if not sample.vega_lite_spec:
                return False
            
            # Feature validation (spec path-op-value constraints)
            if sample.features:
                for f in sample.features:
                    if not self._check_feature(sample.vega_lite_spec, f):
                        logger.debug(f"Feature check failed: {f.path}")
                        # Don't fail entirely, just log
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False
    
    def _check_feature(self, spec: Dict[str, Any], f: Feature) -> bool:
        """Check if a feature is satisfied in the spec."""
        # Navigate path
        parts = f.path.split(".")
        current = spec
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Path not found
                if f.op.value == "exists":
                    return False
                return True  # Other ops: skip if path not found
        
        # Check op
        if f.op.value == "exists":
            return True
        elif f.op.value == "eq":
            return current == f.value
        elif f.op.value == "in":
            return current in f.value if f.value else False
        
        return True
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def run(
        self,
        max_instances: Optional[int] = None,
        num_candidates: int = 5,
        db_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        question_id: Optional[int] = None,
        show_progress: bool = True,
        parallel: Optional[bool] = None
    ) -> List[InstanceResult]:
        """
        Run the pipeline on BIRD Mini-Dev dataset with smart resume.
        Automatically detects and continues unfinished work from samples_full.json.
        
        Args:
            max_instances: Maximum instances to process (None = all)
            num_candidates: Number of candidates per instance
            db_filter: Only process instances for this db_id
            difficulty_filter: Only process instances with this difficulty
            question_id: Process only this specific question_id (for debugging)
            show_progress: Show progress bar
            parallel: Enable parallel processing (None = use config default)
            
        Returns:
            List of InstanceResult
        """
        # Use config default if not specified
        if parallel is None:
            parallel = self.config.pipeline.enable_parallel
        # Initialize state manager
        state_mgr = DatasetStateManager(self.config.paths.output_dir)
        samples_full_path = self.config.paths.output_dir / "samples_full.json"
        statistic_path = self.config.paths.output_dir / "dataset_statistic.json"
        
        # Scan existing samples
        existing_samples = state_mgr.scan_existing_samples(samples_full_path)
        total_existing = sum(len(v) for v in existing_samples.values())
        logger.info(f"Found {total_existing} existing samples from {len(existing_samples)} instances")
        
        # Restore diversity tracker state from samples_full.json
        if samples_full_path.exists():
            try:
                with open(samples_full_path, "r", encoding="utf-8") as f:
                    samples_data = json.load(f)
                
                if samples_data:
                    # Only use validated samples for diversity tracking
                    valid_samples = [s for s in samples_data if s.get("validation_passed", True)]
                    if len(valid_samples) < len(samples_data):
                        logger.info(f"Filtered out {len(samples_data) - len(valid_samples)} invalid samples for diversity tracking")
                    
                    self.diversity_tracker.rebuild_from_samples(valid_samples)
                    logger.info(f"✓ Diversity tracker restored from {len(valid_samples)} samples")
                    logger.info(f"  Overall diversity score: {self.diversity_tracker.get_overall_diversity_score():.3f}")
                    logger.info(f"  Chart types covered: {len(self.diversity_tracker.feature_space.chart_types)}")
            except Exception as e:
                logger.warning(f"Failed to restore diversity tracker: {e}")
        
        # Load instances
        logger.info("Loading BIRD Mini-Dev instances...")
        all_instances = self.preprocessor.load_data_instances()
        
        # Apply filters
        if question_id is not None:
            all_instances = [i for i in all_instances if i.question_id == question_id]
            if not all_instances:
                logger.error(f"No instance found with question_id={question_id}")
                return []
            logger.info(f"Debug mode: Processing only question_id={question_id}")
        
        if db_filter:
            all_instances = [i for i in all_instances if i.db_id == db_filter]
            logger.info(f"Filtered to {len(all_instances)} instances for db_id={db_filter}")
        
        if difficulty_filter:
            all_instances = [i for i in all_instances if i.difficulty.lower() == difficulty_filter.lower()]
            logger.info(f"Filtered to {len(all_instances)} instances with difficulty={difficulty_filter}")
        
        if max_instances and question_id is None:
            all_instances = all_instances[:max_instances]
        
        # Calculate remaining work for each instance
        work_plan = state_mgr.calculate_remaining_work(
            all_instances, num_candidates, existing_samples
        )
        
        total_work = sum(count for _, count in work_plan)
        logger.info(f"Work plan: {len(work_plan)} instances need work, {total_work} candidates to generate")
        
        if total_work == 0:
            logger.info("✓ All work already completed!")
            return []
        
        # Set up signal handlers
        state_mgr.setup_signal_handlers()
        
        # Define samples.json path for incremental saving
        samples_path = self.config.paths.output_dir / "samples.json"
        
        # Process instances: choose between parallel or serial based on config
        try:
            if self.config.pipeline.enable_instance_parallel and parallel:
                # Instance-level parallel processing (NEW)
                logger.info(f"Using instance-level parallel processing (workers={self.config.pipeline.instance_parallel_workers})")
                results = self._process_instances_parallel(
                    work_plan=work_plan,
                    num_candidates=num_candidates,
                    state_mgr=state_mgr,
                    samples_full_path=samples_full_path,
                    samples_path=samples_path,
                    statistic_path=statistic_path,
                    existing_samples=existing_samples,
                    show_progress=show_progress
                )
            else:
                # Serial instance processing (original behavior)
                mode_desc = "serial" if not parallel else "serial instances with parallel candidates"
                logger.info(f"Using {mode_desc} processing")
                results = self._process_instances_serial(
                    work_plan=work_plan,
                    num_candidates=num_candidates,
                    state_mgr=state_mgr,
                    samples_full_path=samples_full_path,
                    samples_path=samples_path,
                    statistic_path=statistic_path,
                    existing_samples=existing_samples,
                    parallel=parallel,
                    show_progress=show_progress
                )
            
        finally:
            # Restore signal handlers
            state_mgr.restore_signal_handlers()
        
        # Generate final outputs
        self._save_final_outputs(results, state_mgr, samples_full_path, statistic_path)
        
        # Summary
        total_attempted = sum(r.candidates_attempted for r in results)
        total_succeeded = sum(r.candidates_succeeded for r in results)
        total_samples = sum(len(r.samples) for r in results)
        
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"  Instances processed: {len(results)}")
        logger.info(f"  Candidates attempted: {total_attempted}")
        logger.info(f"  Candidates succeeded: {total_succeeded}")
        logger.info(f"  Success rate: {total_succeeded / max(total_attempted, 1) * 100:.1f}%")
        logger.info(f"  Total samples generated: {total_samples}")
        
        # Log diversity statistics
        if self._diversity_tracker and self._diversity_tracker.total_samples > 0:
            # Get all possible chart types from key_feature_gen
            all_chart_types = self.key_feature_gen.ALL_CHART_TYPES if self._key_feature_gen else None
            diversity_report = self._diversity_tracker.get_diversity_report(all_chart_types=all_chart_types)
            logger.info(f"  Diversity score: {diversity_report['overall_diversity_score']:.2f}")
            logger.info(f"  Chart types used: {diversity_report['chart_type_stats']['total_chart_types_used']}")
        
        if state_mgr.is_interrupted:
            logger.info("=" * 60)
            logger.info("⚠️  Run was interrupted. Run again to continue.")
        
        logger.info("=" * 60)
        
        return results
    
    # =========================================================================
    # Output
    # =========================================================================
    
    def save_samples(
        self,
        results: List[InstanceResult],
        output_path: Path,
        samples_full_path: Optional[Path] = None,
        format: str = "compact"
    ):
        """
        Save generated samples to JSON file.
        Reads all samples from samples_full.json and saves in requested format.
        
        Args:
            results: List of InstanceResult from current run (for logging)
            output_path: Path to save samples JSON
            samples_full_path: Path to samples_full.json (defaults to config output_dir)
            format: "compact" | "full" | "export"
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if samples_full_path is None:
            samples_full_path = self.config.paths.output_dir / "samples_full.json"
        
        # Read all samples from samples_full.json
        all_samples = []
        if samples_full_path.exists():
            try:
                with open(samples_full_path, "r", encoding="utf-8") as f:
                    all_samples = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read samples_full.json: {e}")
        
        # Filter: only include validated samples in output
        valid_samples = [s for s in all_samples if s.get("validation_passed", True)]
        if len(valid_samples) < len(all_samples):
            logger.info(f"Filtered out {len(all_samples) - len(valid_samples)} invalid samples from output")
        
        # Convert to requested format
        if format == "compact":
            formatted_samples = [self._convert_sample_format(s, "compact") for s in valid_samples]
        elif format == "export":
            formatted_samples = [self._convert_sample_format(s, "export") for s in valid_samples]
        else:
            formatted_samples = valid_samples  # Already in full format
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_samples, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
        
        logger.info(f"Saved {len(formatted_samples)} samples to {output_path} (format: {format})")
    
    def _convert_sample_format(self, sample_dict: Dict[str, Any], format: str) -> Dict[str, Any]:
        """Convert a sample dict from full format to compact or export format."""
        if format == "compact":
            return {
                "sample_id": sample_dict.get("sample_id"),
                "source_question_id": sample_dict.get("source_question_id"),
                "candidate_id": sample_dict.get("candidate_id"),
                "db_id": sample_dict.get("db_id"),
                "data_processing_code": sample_dict.get("data_processing_code"),
                "vis_question": sample_dict.get("vis_question"),
                "vis_question_clear": sample_dict.get("vis_question_clear"),
                "original_question": sample_dict.get("original_question"),
                "altair_code": sample_dict.get("altair_code"),
                "ambiguity_profile": sample_dict.get("ambiguity_profile"),
                "key_features": sample_dict.get("required_key_features", sample_dict.get("key_features", [])),
                "chart_type": sample_dict.get("chart_type"),
                "chart_category": sample_dict.get("chart_category"),
                "difficulty": sample_dict.get("difficulty"),
            }
        elif format == "export":
            return {
                "sample_id": sample_dict.get("sample_id"),
                "source_question_id": sample_dict.get("source_question_id"),
                "candidate_id": sample_dict.get("candidate_id"),
                "vis_sql": sample_dict.get("vis_sql", ""),
                "data_processing_code": sample_dict.get("data_processing_code"),
                "vis_question": sample_dict.get("vis_question"),
                "vis_question_clear": sample_dict.get("vis_question_clear"),
                "original_question": sample_dict.get("original_question"),
                "altair_code": sample_dict.get("altair_code"),
                "ambiguity_profile": sample_dict.get("ambiguity_profile"),
                "features": sample_dict.get("features", []),
                "required_key_features": sample_dict.get("required_key_features", []),
                "chart_type": sample_dict.get("chart_type"),
                "chart_category": sample_dict.get("chart_category"),
                "difficulty": sample_dict.get("difficulty"),
            }
        return sample_dict
    
    def _save_final_outputs(
        self,
        results: List[InstanceResult],
        state_mgr: DatasetStateManager,
        samples_full_path: Path,
        statistic_path: Path
    ):
        """Save final output files: samples.json and dataset_statistic.json."""
        # Save samples.json (compact format)
        samples_path = self.config.paths.output_dir / "samples.json"
        self.save_samples(results, samples_path, samples_full_path, format="compact")
        
        # Build and save final dataset_statistic.json
        final_statistics = self._build_final_statistics(results)
        state_mgr.save_statistics(final_statistics, statistic_path)
        logger.info(f"Saved final statistics to {statistic_path}")
    
    def _build_statistics_incremental(self) -> Dict[str, Any]:
        """Build statistics for incremental update (fast mode)."""
        # Get all possible chart types from key_feature_gen
        all_chart_types = self.key_feature_gen.ALL_CHART_TYPES if self._key_feature_gen else None
        diversity_report = self._diversity_tracker.get_diversity_report(all_chart_types=all_chart_types) if self._diversity_tracker else None
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_samples": self._diversity_tracker.total_samples if self._diversity_tracker else 0
            },
            "diversity": diversity_report
        }
    
    def _build_final_statistics(self, results: List[InstanceResult]) -> Dict[str, Any]:
        """Build complete statistics for final save."""
        # Calculate summary
        summary = self._calculate_summary(results)
        
        # Calculate distribution
        distribution = self._calculate_distribution(results)
        
        # Get diversity report
        diversity_report = None
        if self._diversity_tracker and self._diversity_tracker.total_samples > 0:
            # Get all possible chart types from key_feature_gen
            all_chart_types = self.key_feature_gen.ALL_CHART_TYPES if self._key_feature_gen else None
            diversity_report = self._diversity_tracker.get_diversity_report(all_chart_types=all_chart_types)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "distribution": distribution,
            "diversity": diversity_report,
            "instance_details": [r.to_dict() for r in results]
        }
    
    def _calculate_summary(self, results: List[InstanceResult]) -> Dict[str, Any]:
        """Calculate execution summary statistics."""
        total_instances = len(results)
        total_samples = sum(len(r.samples) for r in results)
        total_attempted = sum(r.candidates_attempted for r in results)
        total_succeeded = sum(r.candidates_succeeded for r in results)
        
        return {
            "instances_processed": total_instances,
            "candidates_attempted": total_attempted,
            "candidates_succeeded": total_succeeded,
            "success_rate": f"{total_succeeded / max(total_attempted, 1) * 100:.1f}%",
            "total_samples": total_samples
        }
    
    def _calculate_distribution(self, results: List[InstanceResult]) -> Dict[str, Dict[str, int]]:
        """Calculate distribution statistics."""
        chart_types = {}
        intent_types = {}
        databases = {}
        difficulties = {}
        
        for result in results:
            db_id = result.instance.db_id
            databases[db_id] = databases.get(db_id, 0) + len(result.samples)
            
            for sample in result.samples:
                ct = sample.chart_type
                chart_types[ct] = chart_types.get(ct, 0) + 1
                
                it = sample.vis_intent_type
                intent_types[it] = intent_types.get(it, 0) + 1
                
                diff = sample.difficulty
                difficulties[diff] = difficulties.get(diff, 0) + 1
        
        return {
            "chart_types": chart_types,
            "intent_types": intent_types,
            "databases": databases,
            "difficulties": difficulties
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.preprocessor.get_stats()


def create_pipeline(project_root: Optional[Path] = None) -> Pipeline:
    """Create and initialize pipeline."""
    if project_root:
        init_config(project_root)
    
    config = get_config()
    config.ensure_dirs()
    
    return Pipeline()
