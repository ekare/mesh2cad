from __future__ import annotations

import sys
import time

import pytest

from mesh2cad.learning import prepare
from mesh2cad.learning.prepare import (
    _append_failure,
    _load_failure_ledger,
    _parallel_batch_results,
    _prepare_step_job,
    _select_step_files,
    prepare_step_directory,
)


def test_random_step_selection_is_deterministic_and_not_a_prefix(tmp_path) -> None:
    files = []
    for index in range(20):
        path = tmp_path / f"part-{index:02d}.step"
        path.touch()
        files.append(path)

    first, available = _select_step_files(tmp_path, limit=None, sample_count=6, seed=23)
    second, _ = _select_step_files(tmp_path, limit=None, sample_count=6, seed=23)

    assert available == 20
    assert first == second
    assert first != files[:6]


def test_step_selection_rejects_conflicting_staging_options(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_step_files(tmp_path, limit=1, sample_count=1, seed=17)


def test_full_selection_can_be_shuffled_reproducibly(tmp_path) -> None:
    ordered = []
    for index in range(20):
        path = tmp_path / f"part-{index:02d}.step"
        path.touch()
        ordered.append(path)

    first, _ = _select_step_files(
        tmp_path, limit=None, sample_count=None, seed=41, shuffle=True
    )
    second, _ = _select_step_files(
        tmp_path, limit=None, sample_count=None, seed=41, shuffle=True
    )

    assert first == second
    assert first != ordered
    assert set(first) == set(ordered)


def test_step_job_timeout_is_reported_as_one_model_failure(tmp_path, monkeypatch) -> None:
    source = tmp_path / "slow.step"
    source.touch()

    def slow_conversion(*_args, **_kwargs):
        time.sleep(1.0)

    monkeypatch.setattr(prepare, "step_to_sample", slow_conversion)
    result = _prepare_step_job(
        (str(source), str(tmp_path / "slow.npz"), None, 0.001, 0.1, None, 0.01)
    )

    assert result["status"] == "failed"
    assert "TimeoutError" in result["error"]
    assert not (tmp_path / "slow.npz").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="hard watchdog runs in the WSL/POSIX path")
def test_native_worker_is_killed_when_python_signal_cannot_interrupt(tmp_path, monkeypatch) -> None:
    sources = [tmp_path / "slow-a.step", tmp_path / "slow-b.step"]
    for source in sources:
        source.touch()

    def uninterruptible_conversion(*_args, **_kwargs):
        signal_module = prepare.signal
        signal_module.signal(signal_module.SIGALRM, signal_module.SIG_IGN)
        time.sleep(5.0)

    monkeypatch.setattr(prepare, "step_to_sample", uninterruptible_conversion)
    jobs = [
        (str(source), str(source.with_suffix(".npz")), None, 0.001, 0.1, None, None)
        for source in sources
    ]

    started = time.monotonic()
    results = list(_parallel_batch_results(jobs, workers=2, hard_timeout_seconds=0.1))

    assert time.monotonic() - started < 3.0
    assert len(results) == 2
    assert all(result["status"] == "failed" for result in results)
    assert all("was terminated" in result["error"] for result in results)


@pytest.mark.skipif(sys.platform == "win32", reason="memory watchdog runs in WSL/POSIX")
def test_worker_pool_is_killed_before_system_memory_is_exhausted(tmp_path, monkeypatch) -> None:
    sources = [tmp_path / "large-a.step", tmp_path / "large-b.step"]
    for source in sources:
        source.touch()

    def slow_conversion(*_args, **_kwargs):
        time.sleep(5.0)

    monkeypatch.setattr(prepare, "step_to_sample", slow_conversion)
    monkeypatch.setattr(prepare, "_available_memory_bytes", lambda: 0)
    jobs = [
        (str(source), str(source.with_suffix(".npz")), None, 0.001, 0.1, None, None)
        for source in sources
    ]

    started = time.monotonic()
    results = list(
        _parallel_batch_results(
            jobs,
            workers=2,
            hard_timeout_seconds=None,
            min_available_memory_bytes=1,
        )
    )

    assert time.monotonic() - started < 3.0
    assert len(results) == 2
    assert all(result["status"] == "failed" for result in results)
    assert all("MemoryLimitError" in result["error"] for result in results)


def test_failure_ledger_migrates_and_skips_rejected_paths_on_resume(tmp_path) -> None:
    source_root = tmp_path / "step"
    target = tmp_path / "prepared"
    source_root.mkdir()
    target.mkdir()
    source = source_root / "invalid.step"
    source.touch()
    migrated = {"path": str(source), "error": "CadLabelError: invalid BRep"}
    (target / "manifest.partial.json").write_text(
        prepare.json.dumps({"failures": [migrated]}), encoding="utf-8"
    )

    ledger_path, failures = _load_failure_ledger(target)
    extra = {"path": str(source_root / "other.step"), "error": "TimeoutError: terminated"}
    _append_failure(ledger_path, extra)
    _ledger_path, reloaded = _load_failure_ledger(target)
    summary = prepare_step_directory(source_root, target)

    assert failures == {str(source): migrated}
    assert set(reloaded) == {str(source), extra["path"]}
    assert summary["processed"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["skipped_failed_existing"] == 1
