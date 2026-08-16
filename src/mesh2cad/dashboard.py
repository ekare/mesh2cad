"""Asynchronous local telemetry collector for preparation and training."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _snapshot_is_fresh(snapshot: dict[str, Any] | None, *, max_age_seconds: float = 60.0) -> bool:
    if not snapshot or snapshot.get("status") != "running":
        return False
    updated_at = snapshot.get("updated_at")
    if not isinstance(updated_at, str):
        return False
    try:
        timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    return -5.0 <= age <= max_age_seconds


def _snapshot_age_seconds(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot or not isinstance(snapshot.get("updated_at"), str):
        return None
    try:
        timestamp = datetime.fromisoformat(snapshot["updated_at"].replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


async def _run_command(*command: str, timeout: float = 2.0) -> tuple[int, str]:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return 127, ""
    try:
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        return 124, ""
    return int(process.returncode or 0), stdout.decode("utf-8", errors="replace").strip()


async def _service_snapshot(unit: str) -> dict[str, str]:
    code, output = await _run_command(
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=ActiveState",
        "--property=SubState",
        "--property=ActiveEnterTimestamp",
        "--no-pager",
    )
    if code != 0:
        return {"active_state": "unknown", "sub_state": "unknown"}
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "active_since": values.get("ActiveEnterTimestamp", ""),
    }


async def _gpu_snapshot(gpu_uuid: str | None) -> dict[str, Any] | None:
    command = ["nvidia-smi"]
    if gpu_uuid:
        command.append(f"--id={gpu_uuid}")
    command.extend(
        (
            "--query-gpu=name,pci.bus_id,uuid,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        )
    )
    code, output = await _run_command(*command)
    if code != 0 or not output:
        return None
    fields = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(fields) != 9:
        return None
    numeric_names = (
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
        "temperature_c",
        "power_w",
        "power_limit_w",
    )
    numeric_values: dict[str, float | None] = {}
    for name, value in zip(numeric_names, fields[3:], strict=True):
        try:
            numeric_values[name] = float(value)
        except ValueError:
            numeric_values[name] = None
    return {
        "name": fields[0],
        "pci_bus_id": fields[1],
        "uuid": fields[2],
        **numeric_values,
    }


async def collect_dashboard_snapshot(
    state_root: str | Path,
    *,
    gpu_uuid: str | None = None,
) -> dict[str, Any]:
    root = Path(state_root)
    preparation = _read_snapshot(root / "preparation.json")
    training = _read_snapshot(root / "training.json")
    preparation_service, training_service, gpu = await asyncio.gather(
        _service_snapshot("mesh2cad-prepare.service"),
        _service_snapshot("mesh2cad-train.service"),
        _gpu_snapshot(gpu_uuid),
    )
    training_age = _snapshot_age_seconds(training)
    if training is not None:
        training["telemetry_age_seconds"] = training_age

    if training_service["active_state"] == "active" or _snapshot_is_fresh(training):
        preparing_large_batch = (
            training_service["active_state"] == "active"
            and training is not None
            and training.get("phase") == "training"
            and training_age is not None
            and training_age > 15.0
        )
        operation = {
            "kind": "training",
            "label": (
                "Preparing large mesh batch" if preparing_large_batch else "Neural training"
            ),
            "state": "running",
        }
    elif preparation_service["active_state"] == "active" or _snapshot_is_fresh(preparation):
        operation = {
            "kind": "preparation",
            "label": "STEP dataset preparation",
            "state": "running",
        }
    elif training and training.get("status") == "failed":
        operation = {"kind": "training", "label": "Training needs attention", "state": "failed"}
    else:
        operation = {"kind": "idle", "label": "Ready", "state": "idle"}

    storage = None
    try:
        usage = shutil.disk_usage(root)
        storage = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError:
        pass

    return {
        "server_time": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "preparation": preparation,
        "training": training,
        "gpu": gpu,
        "storage": storage,
        "services": {
            "preparation": preparation_service,
            "training": training_service,
        },
    }
