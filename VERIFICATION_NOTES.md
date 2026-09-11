# Verification Notes - v0.1.0

Generated during the final verification pass on 2026-05-07.

## Environment

- OS: Linux (Pop!_OS 24.04, kernel 6.18.7)
- CPU: Intel Core i5-10400F @ 2.90GHz
- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM, 11 GB free)
- Python: 3.13.12
- PyTorch: 2.11.0+cu130

## Fixes Applied During Verification

### Fix 1: Batch mode - pickling error (nested worker function)

**File:** `pureframe/batch.py`  
**Error:** `AttributeError: Can't get local object 'process_folder.<locals>.worker'`  
**Cause:** The `worker` function was defined inside `process_folder`, making it unpicklable by Python's multiprocessing.  
**Fix:** Moved `worker` → `_batch_worker` at module level.

### Fix 2: Batch mode - profile=None in subprocesses

**File:** `pureframe/batch.py`  
**Error:** `1 validation error for ProfileSettings: profile - Input should be 'HIGH', 'MEDIUM', 'LOW' or 'CPU'`  
**Cause:** `base_config.profile` could be `None` when no `--profile` is passed via CLI; this None propagated into per-file configs passed to subprocesses.  
**Fix:** Added `resolved_profile = base_config.profile or detect_profile()` before the config clone loop.

### Fix 3: Batch mode - CUDA re-initialization in forked subprocesses

**File:** `pureframe/batch.py`  
**Error:** `Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method`  
**Cause:** Linux default multiprocessing start method is `fork`, which cannot re-initialize CUDA in child processes.  
**Fix:** Changed `ProcessPoolExecutor` to use `mp_context=multiprocessing.get_context("spawn")`.

### Fix 4: Test suite broken by mp_context kwarg

**File:** `tests/test_batch.py`  
**Error:** `TypeError: DummyExecutor.__init__() got an unexpected keyword argument 'mp_context'`  
**Cause:** Test's `DummyExecutor` mock didn't accept the new `mp_context` kwarg added in Fix 3.  
**Fix:** Added `mp_context=None` to `DummyExecutor.__init__`.

## Resumability Status

The checkpoint system correctly detects already-completed jobs (via SQLite job state + config hash) and skips them. If you use the same input path but a different output path, the system recognises the job as DONE and will not re-process it (since jobs are keyed on input path + config hash, not output path). This is by design - to force a re-process with a different output path, use `pureframe jobs cleanup <job_id>` first.

## Known GitHub Issues to Create After Launch

1. Resumability: interrupted jobs should support different output paths on resume.
2. `pureframe jobs cleanup --all` flag not implemented - only individual job cleanup supported.
3. CPU profile timing appears faster than MEDIUM/HIGH on zero-detection synthetic content due to CUDA overhead amortisation. Add a note in docs.
