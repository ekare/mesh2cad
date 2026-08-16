import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import httpx

from mesh2cad.synthetic import write_cube_with_hole
from mesh2cad.webapp import ConversionStore, _parser, _safe_output_name, create_app


def test_output_name_is_sanitized():
    assert _safe_output_name("../../my unsafe part.stl") == "myunsafepart.step"


def test_service_launcher_can_require_a_stable_port():
    args = _parser().parse_args(["--port", "7860", "--strict-port", "--no-browser"])

    assert args.port == 7860
    assert args.strict_port is True
    assert args.no_browser is True


def test_web_api_converts_and_downloads_step(tmp_path: Path):
    async def exercise_api() -> None:
        source = write_cube_with_hole(tmp_path / "source.stl", sections_per_side=8)
        store = ConversionStore(root=tmp_path / "jobs")
        transport = httpx.ASGITransport(app=create_app(store=store))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert page.headers["cache-control"] == "no-store"
            assert "From mesh" in page.text
            assert "Reconstructed STEP" in page.text
            assert "Deviation heatmap" in page.text
            assert 'src="/app.js?' in page.text
            assert "Prepared together with ChatGPT" in page.text

            script = await client.get("/app.js")
            assert script.status_code == 200
            assert 'fetch("/api/convert"' in script.text
            assert "showReconstruction" in script.text
            assert "geometric_fidelity" in script.text
            assert "mesh_quality" in script.text

            with source.open("rb") as stream:
                response = await client.post(
                    "/api/convert", files={"file": ("demo.stl", stream, "model/stl")}
                )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["surfaces"] == {
                "engine": "deterministic",
                "planes": 6,
                "cylinders": 1,
                "unassigned_faces": 0,
            }
            assert payload["topology"]["valid"] is True
            assert payload["preview_error"] is None
            assert payload["preview"]["total_triangles"] > 0
            assert payload["preview"]["shown_triangles"] > 0
            assert len(payload["preview"]["positions"]) % 9 == 0
            assert len(payload["preview"]["deviations"]) == len(
                payload["preview"]["positions"]
            ) // 9
            assert payload["metrics"]["geometric_fidelity"] > 0.8
            assert payload["metrics"]["bidirectional_p95_error"] >= 0.0
            assert payload["cad_quality"]["representation"] == "analytic-csg"
            assert payload["cad_quality"]["brep_face_count"] < payload["mesh"]["faces"]
            assert payload["mesh_quality"]["watertight"] is True
            assert payload["mesh_quality"]["repair_recommended"] is False

            download = await client.get(payload["download_url"])
            assert download.status_code == 200
            assert b"ISO-10303-21" in download.content[:100]
        store.close()

    asyncio.run(exercise_api())
