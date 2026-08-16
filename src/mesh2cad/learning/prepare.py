"""Prepare labeled triangle-graph datasets from STEP or synthetic meshes."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import shutil
import signal
import sys
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np

from mesh2cad.learning.cad_labels import CadLabelError, step_to_sample
from mesh2cad.learning.features import SURFACE_TO_ID, build_sample, save_sample
from mesh2cad.learning.telemetry import publish_telemetry
from mesh2cad.mesh import prepare_mesh
from mesh2cad.segmentation import segment_surfaces
from mesh2cad.synthetic import cube_with_hole


def _prepare_step_job(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        step_path_value,
        output_value,
        linear_deflection,
        relative_deflection,
        angular_deflection,
        max_triangles,
        job_timeout_seconds,
    ) = job
    step_path = Path(step_path_value)
    output = Path(output_value)
    if output.exists():
        return {"status": "completed", "path": str(step_path), "skipped": True}

    alarm_supported = all(
        hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")
    )
    if job_timeout_seconds is not None and not alarm_supported:
        return {
            "status": "failed",
            "path": str(step_path),
            "error": "TimeoutError: per-model STEP timeouts require a POSIX worker",
        }

    previous_alarm_handler: Any = None

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"STEP preparation exceeded {job_timeout_seconds:g} seconds")

    try:
        if job_timeout_seconds is not None:
            previous_alarm_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, job_timeout_seconds)
        sample = step_to_sample(
            step_path,
            linear_deflection=linear_deflection,
            relative_deflection=relative_deflection,
            angular_deflection=angular_deflection,
        )
        triangle_count = len(sample.faces)
        if max_triangles is not None and triangle_count > max_triangles:
            raise CadLabelError(f"triangle cap exceeded: {triangle_count} > {max_triangles}")
        save_sample(sample, output)
        return {
            "status": "completed",
            "path": str(step_path),
            "triangles": triangle_count,
            "vertices": len(sample.vertices),
            "instances": len(np.unique(sample.face_instance_labels)),
            "skipped": False,
        }
    except (CadLabelError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return {
            "status": "failed",
            "path": str(step_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if job_timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)


def _select_step_files(
    source: Path,
    *,
    limit: int | None,
    sample_count: int | None,
    seed: int,
    shuffle: bool = False,
) -> tuple[list[Path], int]:
    files = sorted((*source.rglob("*.step"), *source.rglob("*.stp")))
    available = len(files)
    if limit is not None and sample_count is not None:
        raise ValueError("limit and sample_count are mutually exclusive")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        files = files[:limit]
    elif sample_count is not None:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if sample_count < available:
            files = sorted(random.Random(seed).sample(files, sample_count))
    if shuffle:
        random.Random(seed).shuffle(files)
    return files, available


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_failure_ledger(target: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    ledger_path = target / "failures.jsonl"
    failures: dict[str, dict[str, str]] = {}
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "path" in record and "error" in record:
                failures[str(record["path"])] = {
                    "path": str(record["path"]),
                    "error": str(record["error"]),
                }

    # Migrate failures written before the append-only ledger was introduced.
    for manifest_path in (target / "manifest.partial.json", target / "manifest.json"):
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in manifest.get("failures", []):
            if isinstance(record, dict) and "path" in record and "error" in record:
                failures.setdefault(
                    str(record["path"]),
                    {"path": str(record["path"]), "error": str(record["error"])},
                )

    temporary = ledger_path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in failures.values():
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(ledger_path)
    return ledger_path, failures


def _append_failure(ledger_path: Path, record: dict[str, str]) -> None:
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _available_memory_bytes() -> int | None:
    """Return Linux MemAvailable without adding a runtime dependency."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _parallel_batch_results(
    batch: Sequence[tuple[Any, ...]],
    *,
    workers: int,
    hard_timeout_seconds: float | None,
    min_available_memory_bytes: int = 0,
) -> Iterator[dict[str, Any]]:
    """Run bounded CAD jobs, terminating native workers before timeout or system OOM."""
    remaining = deque(batch)
    context_name = "spawn" if sys.platform == "win32" else "fork"
    context = multiprocessing.get_context(context_name)

    while remaining:
        pool = context.Pool(processes=workers)
        pending: list[tuple[tuple[Any, ...], Any, float]] = []

        def fill_window(
            active: list[tuple[tuple[Any, ...], Any, float]] = pending,
            worker_pool: Any = pool,
        ) -> None:
            while remaining and len(active) < workers:
                job = remaining.popleft()
                active.append(
                    (job, worker_pool.apply_async(_prepare_step_job, (job,)), monotonic())
                )

        fill_window()
        restart_pool = False
        try:
            while pending:
                ready = [entry for entry in pending if entry[1].ready()]
                if ready:
                    for entry in ready:
                        pending.remove(entry)
                        job, asynchronous_result, _submitted_at = entry
                        try:
                            yield asynchronous_result.get()
                        except Exception as exc:
                            yield {
                                "status": "failed",
                                "path": job[0],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                    fill_window()
                    continue

                now = monotonic()
                timed_out = (
                    []
                    if hard_timeout_seconds is None
                    else [
                        entry
                        for entry in pending
                        if now - entry[2] >= hard_timeout_seconds
                    ]
                )
                if timed_out:
                    retry = [entry[0] for entry in pending if entry not in timed_out]
                    for job in reversed(retry):
                        remaining.appendleft(job)
                    pool.terminate()
                    pool.join()
                    for job, _asynchronous_result, _submitted_at in timed_out:
                        yield {
                            "status": "failed",
                            "path": job[0],
                            "error": (
                                "TimeoutError: native CAD worker exceeded "
                                f"{hard_timeout_seconds:g} seconds and was terminated"
                            ),
                        }
                    restart_pool = True
                    break

                available_memory = _available_memory_bytes()
                if (
                    min_available_memory_bytes > 0
                    and available_memory is not None
                    and available_memory < min_available_memory_bytes
                ):
                    interrupted = [entry[0] for entry in pending]
                    pool.terminate()
                    pool.join()
                    for job in interrupted:
                        yield {
                            "status": "failed",
                            "path": job[0],
                            "error": (
                                "MemoryLimitError: native CAD worker pool was terminated "
                                f"with {available_memory / 1024**3:.2f} GiB available "
                                f"(reserve: {min_available_memory_bytes / 1024**3:.2f} GiB)"
                            ),
                        }
                    restart_pool = True
                    break
                sleep(0.1)
        except BaseException:
            pool.terminate()
            pool.join()
            raise

        if restart_pool:
            continue
        pool.close()
        pool.join()


def prepare_step_directory(
    source: str | Path,
    destination: str | Path,
    *,
    limit: int | None = None,
    sample_count: int | None = None,
    seed: int = 17,
    shuffle: bool = False,
    linear_deflection: float | None = None,
    relative_deflection: float = 2e-3,
    angular_deflection: float = 0.2,
    max_triangles: int | None = None,
    workers: int = 1,
    progress_every: int = 50,
    job_timeout_seconds: float | None = None,
    min_free_gib: float = 0.0,
    min_available_memory_gib: float = 0.0,
    pool_batch_size: int = 512,
    retry_failures: bool = False,
    telemetry_path: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source)
    target = Path(destination)
    files, available = _select_step_files(
        source_path,
        limit=limit,
        sample_count=sample_count,
        seed=seed,
        shuffle=shuffle,
    )
    target.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if pool_batch_size < 1:
        raise ValueError("pool_batch_size must be at least 1")
    if min_free_gib < 0.0:
        raise ValueError("min_free_gib cannot be negative")
    if min_available_memory_gib < 0.0:
        raise ValueError("min_available_memory_gib cannot be negative")
    if job_timeout_seconds is not None:
        if job_timeout_seconds <= 0.0:
            raise ValueError("job_timeout_seconds must be positive")
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
            raise ValueError("per-model timeouts require a POSIX environment")

    ledger_path, known_failures = _load_failure_ledger(target)
    jobs: list[tuple[Any, ...]] = []
    skipped_existing = 0
    skipped_failed_existing = 0
    retained_failures: list[dict[str, str]] = []
    for step_path in files:
        output = target / f"{step_path.stem}.npz"
        if output.exists():
            skipped_existing += 1
            continue
        previous_failure = known_failures.get(str(step_path))
        if previous_failure is not None and not retry_failures:
            skipped_failed_existing += 1
            retained_failures.append(previous_failure)
            continue
        jobs.append(
            (
                str(step_path),
                str(output),
                linear_deflection,
                relative_deflection,
                angular_deflection,
                max_triangles,
                job_timeout_seconds,
            )
        )

    completed = skipped_existing
    skipped = skipped_existing
    failures: list[dict[str, str]] = retained_failures
    triangle_counts: list[int] = []
    processed = skipped_existing + skipped_failed_existing
    storage_limit_reached = False
    reserve_bytes = int(min_free_gib * 1024**3)
    memory_reserve_bytes = int(min_available_memory_gib * 1024**3)
    progress_path = target / "manifest.partial.json"
    initial_processed = processed

    def publish_preparation_state(status: str) -> None:
        elapsed = max(perf_counter() - started, 1e-9)
        completed_this_run = max(0, processed - initial_processed)
        rate_per_minute = completed_this_run / elapsed * 60.0
        remaining = len(files) - processed
        publish_telemetry(
            telemetry_path,
            {
                "status": status,
                "phase": "preparation",
                "available": available,
                "selected": len(files),
                "processed": processed,
                "completed": completed,
                "failed": len(failures),
                "remaining": remaining,
                "progress_fraction": processed / max(1, len(files)),
                "rate_per_minute": rate_per_minute,
                "eta_seconds": (
                    remaining / rate_per_minute * 60.0 if rate_per_minute > 0 else None
                ),
                "free_bytes": shutil.disk_usage(target).free,
                "workers": workers,
                "elapsed_seconds": elapsed,
            },
        )

    publish_preparation_state("running")

    def record_result(result: dict[str, Any]) -> None:
        nonlocal completed, processed, skipped
        processed += 1
        if result["status"] == "completed":
            completed += 1
            skipped += int(result.get("skipped", False))
            if "triangles" in result:
                triangle_counts.append(int(result["triangles"]))
        else:
            record = {"path": result["path"], "error": result["error"]}
            failures.append(record)
            if record["path"] not in known_failures:
                known_failures[record["path"]] = record
                _append_failure(ledger_path, record)
        if progress_every > 0 and (processed % progress_every == 0 or processed == len(files)):
            publish_preparation_state("running")
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "completed": completed,
                        "failed": len(failures),
                        "remaining": len(files) - processed,
                        "free_gib": round(shutil.disk_usage(target).free / 1024**3, 2),
                    }
                ),
                flush=True,
            )

    for offset in range(0, len(jobs), pool_batch_size):
        free_bytes = shutil.disk_usage(target).free
        if free_bytes <= reserve_bytes:
            storage_limit_reached = True
            break
        batch = jobs[offset : offset + pool_batch_size]
        if workers == 1:
            for job in batch:
                record_result(_prepare_step_job(job))
        else:
            for result in _parallel_batch_results(
                batch,
                workers=workers,
                hard_timeout_seconds=job_timeout_seconds,
                min_available_memory_bytes=memory_reserve_bytes,
            ):
                record_result(result)

        _write_json_atomic(
            progress_path,
            {
                "status": "running",
                "available": available,
                "selected": len(files),
                "processed": processed,
                "completed": completed,
                "failed": len(failures),
                "skipped_existing": skipped,
                "skipped_failed_existing": skipped_failed_existing,
                "remaining": len(files) - processed,
                "free_bytes": shutil.disk_usage(target).free,
                "failures": failures,
            },
        )

    summary = {
        "status": "storage_limit" if storage_limit_reached else "complete",
        "available": available,
        "discovered": len(files),
        "processed": processed,
        "remaining": len(files) - processed,
        "completed": completed,
        "failed": len(failures),
        "skipped_existing": skipped,
        "skipped_failed_existing": skipped_failed_existing,
        "duration_seconds": perf_counter() - started,
        "configuration": {
            "linear_deflection": linear_deflection,
            "relative_deflection": relative_deflection,
            "angular_deflection": angular_deflection,
            "max_triangles": max_triangles,
            "workers": workers,
            "selection_limit": limit,
            "sample_count": sample_count,
            "seed": seed,
            "shuffle": shuffle,
            "job_timeout_seconds": job_timeout_seconds,
            "min_free_gib": min_free_gib,
            "min_available_memory_gib": min_available_memory_gib,
            "pool_batch_size": pool_batch_size,
            "retry_failures": retry_failures,
        },
        "generated_triangle_count": {
            "total": sum(triangle_counts),
            "min": min(triangle_counts, default=None),
            "max": max(triangle_counts, default=None),
        },
        "failures": failures,
    }
    _write_json_atomic(target / "manifest.json", summary)
    _write_json_atomic(progress_path, summary)
    publish_preparation_state(summary["status"])
    return summary


def prepare_synthetic_dataset(
    destination: str | Path,
    *,
    count: int = 32,
    seed: int = 17,
) -> dict[str, int]:
    """Create a small labeled corpus for CI and neural-pipeline smoke tests."""
    target = Path(destination)
    randomizer = random.Random(seed)
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        size = (
            randomizer.uniform(7.0, 14.0),
            randomizer.uniform(7.0, 14.0),
            randomizer.uniform(3.0, 9.0),
        )
        radius = randomizer.uniform(0.8, 0.35 * min(size[0], size[1]))
        sections = randomizer.randint(8, 20)
        mesh = cube_with_hole(size=size, radius=radius, sections_per_side=sections)
        data = prepare_mesh(mesh)
        segmentation = segment_surfaces(data)
        surface_labels = np.full(len(data.mesh.faces), SURFACE_TO_ID["other"], dtype=np.int64)
        instance_labels = np.full(len(data.mesh.faces), -1, dtype=np.int64)
        for region in segmentation.regions:
            surface_labels[region.face_indices] = SURFACE_TO_ID[region.kind]
            instance_labels[region.face_indices] = region.id
        if np.any(instance_labels < 0):
            raise RuntimeError("Synthetic segmentation did not label every triangle")
        save_sample(
            build_sample(data.mesh, surface_labels, instance_labels),
            target / f"synthetic_{index:05d}.npz",
        )
    summary = {"completed": count, "failed": 0}
    (target / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create mesh2cad neural training samples")
    subparsers = parser.add_subparsers(dest="command", required=True)
    step = subparsers.add_parser("step", help="tessellate STEP files with BRep face labels")
    step.add_argument("source", type=Path)
    step.add_argument("destination", type=Path)
    step.add_argument("--limit", type=int)
    step.add_argument(
        "--sample-count",
        type=int,
        help="deterministic random staging sample; mutually exclusive with --limit",
    )
    step.add_argument("--seed", type=int, default=17)
    step.add_argument(
        "--shuffle",
        action="store_true",
        help="process the selected corpus in a deterministic shuffled order",
    )
    step.add_argument("--linear-deflection", type=float)
    step.add_argument("--relative-deflection", type=float, default=2e-3)
    step.add_argument("--angular-deflection", type=float, default=0.2)
    step.add_argument(
        "--max-triangles",
        type=int,
        help="optional safety cap; omitted by default so no model is rejected for size",
    )
    step.add_argument("--workers", type=int, default=1)
    step.add_argument("--progress-every", type=int, default=50)
    step.add_argument(
        "--job-timeout-seconds",
        type=float,
        help="POSIX wall-clock timeout for one STEP file; does not cap mesh size",
    )
    step.add_argument(
        "--min-free-gib",
        type=float,
        default=0.0,
        help="stop cleanly before starting another batch when free space reaches this reserve",
    )
    step.add_argument(
        "--min-available-memory-gib",
        type=float,
        default=0.0,
        help="terminate the active worker window before Linux reaches this memory reserve",
    )
    step.add_argument(
        "--pool-batch-size",
        type=int,
        default=512,
        help="jobs per worker-pool generation; recycling bounds native CAD memory growth",
    )
    step.add_argument(
        "--retry-failures",
        action="store_true",
        help="explicitly retry paths recorded in the persistent failure ledger",
    )
    step.add_argument(
        "--telemetry-path",
        type=Path,
        help="optional atomic JSON status file for a live operations dashboard",
    )
    synthetic = subparsers.add_parser("synthetic", help="make a tiny labeled smoke-test dataset")
    synthetic.add_argument("destination", type=Path)
    synthetic.add_argument("--count", type=int, default=32)
    synthetic.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "step":
            summary = prepare_step_directory(
                args.source,
                args.destination,
                limit=args.limit,
                sample_count=args.sample_count,
                seed=args.seed,
                shuffle=args.shuffle,
                linear_deflection=args.linear_deflection,
                relative_deflection=args.relative_deflection,
                angular_deflection=args.angular_deflection,
                max_triangles=args.max_triangles,
                workers=args.workers,
                progress_every=args.progress_every,
                job_timeout_seconds=args.job_timeout_seconds,
                min_free_gib=args.min_free_gib,
                min_available_memory_gib=args.min_available_memory_gib,
                pool_batch_size=args.pool_batch_size,
                retry_failures=args.retry_failures,
                telemetry_path=args.telemetry_path,
            )
        else:
            summary = prepare_synthetic_dataset(args.destination, count=args.count, seed=args.seed)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-dataset: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
