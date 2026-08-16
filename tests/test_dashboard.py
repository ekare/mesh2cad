from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
import httpx

from mesh2cad import dashboard
from mesh2cad.learning.telemetry import publish_telemetry
from mesh2cad.webapp import ConversionStore, create_app


def test_publish_telemetry_writes_replace_safe_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "training.json"

    assert publish_telemetry(destination, {"status": "running", "current_loss": 1.25})
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["status"] == "running"
    assert payload["current_loss"] == 1.25
    assert "updated_at" in payload
    assert not destination.with_suffix(".json.part").exists()


def test_dashboard_page_and_snapshot_use_async_machine_telemetry(tmp_path, monkeypatch) -> None:
    async def fake_service(unit: str) -> dict[str, str]:
        state = "active" if unit == "mesh2cad-train.service" else "inactive"
        return {"active_state": state, "sub_state": "running" if state == "active" else "dead"}

    async def fake_gpu(_uuid: str | None) -> dict[str, object]:
        return {
            "name": "NVIDIA GeForce RTX 4060 Ti",
            "pci_bus_id": "00000000:07:00.0",
            "uuid": "GPU-test",
            "memory_used_mib": 8192.0,
            "memory_total_mib": 16380.0,
            "utilization_percent": 95.0,
            "temperature_c": 54.0,
            "power_w": 120.0,
            "power_limit_w": 165.0,
        }

    monkeypatch.setattr(dashboard, "_service_snapshot", fake_service)
    monkeypatch.setattr(dashboard, "_gpu_snapshot", fake_gpu)
    publish_telemetry(
        tmp_path / "training.json",
        {
            "status": "running",
            "phase": "training",
            "epoch": 2,
            "epochs": 8,
            "current_loss": 0.75,
            "best_validation": {
                "surface_macro_iou": 0.5,
                "supported_surface_macro_iou": 0.8,
                "boundary_f1": 0.9,
            },
            "recent_losses": [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.75}],
        },
    )

    async def exercise() -> None:
        store = ConversionStore(root=tmp_path / "jobs")
        app = create_app(
            store=store,
            dashboard_state_root=tmp_path,
            dashboard_gpu_uuid="GPU-test",
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            page = await client.get("/dashboard")
            script = await client.get("/dashboard.js")
            snapshot = await client.get("/api/dashboard")

        assert page.status_code == 200
        assert "Training Observatory" in page.text
        assert "Know what the" in page.text
        assert "Plane/cylinder IoU" in page.text
        assert 'EventSource("/api/dashboard/stream")' in script.text
        assert "supported_surface_macro_iou" in script.text
        payload = snapshot.json()
        assert payload["operation"]["kind"] == "training"
        assert payload["training"]["current_loss"] == 0.75
        assert payload["gpu"]["name"] == "NVIDIA GeForce RTX 4060 Ti"
        store.close()

    asyncio.run(exercise())


def test_fresh_telemetry_marks_direct_training_as_active(tmp_path, monkeypatch) -> None:
    async def inactive_service(_unit: str) -> dict[str, str]:
        return {"active_state": "inactive", "sub_state": "dead"}

    async def no_gpu(_uuid: str | None) -> None:
        return None

    monkeypatch.setattr(dashboard, "_service_snapshot", inactive_service)
    monkeypatch.setattr(dashboard, "_gpu_snapshot", no_gpu)
    publish_telemetry(tmp_path / "training.json", {"status": "running", "phase": "training"})

    snapshot = asyncio.run(dashboard.collect_dashboard_snapshot(tmp_path))

    assert snapshot["operation"] == {
        "kind": "training",
        "label": "Neural training",
        "state": "running",
    }


def test_active_training_labels_a_long_uncapped_batch_preparation(tmp_path, monkeypatch) -> None:
    async def service(unit: str) -> dict[str, str]:
        state = "active" if unit == "mesh2cad-train.service" else "inactive"
        return {"active_state": state, "sub_state": "running" if state == "active" else "dead"}

    async def no_gpu(_uuid: str | None) -> None:
        return None

    monkeypatch.setattr(dashboard, "_service_snapshot", service)
    monkeypatch.setattr(dashboard, "_gpu_snapshot", no_gpu)
    (tmp_path / "training.json").write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "training",
                "updated_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    snapshot = asyncio.run(dashboard.collect_dashboard_snapshot(tmp_path))

    assert snapshot["operation"] == {
        "kind": "training",
        "label": "Preparing large mesh batch",
        "state": "running",
    }
    assert snapshot["training"]["telemetry_age_seconds"] >= 29.0
