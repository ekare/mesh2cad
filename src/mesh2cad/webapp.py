"""Local-only web inference application and browser launcher."""

import argparse
import asyncio
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Annotated

from mesh2cad.dashboard import collect_dashboard_snapshot
from mesh2cad.mesh import load_mesh
from mesh2cad.metrics import preview_deviations
from mesh2cad.model_resources import default_checkpoint, default_segmentation_engine
from mesh2cad.pipeline import SegmentationMode, convert
from mesh2cad.preview import ShapePreview, tessellate_shape
from mesh2cad.types import ConversionResult


@dataclass(frozen=True)
class StoredConversion:
    directory: Path
    step_path: Path
    created_at: float


class ConversionStore:
    """Short-lived output store; input meshes and STEP files never leave the machine."""

    def __init__(self, *, ttl_seconds: int = 3600, root: Path | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self.root = root or Path(tempfile.mkdtemp(prefix="mesh2cad-web-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, StoredConversion] = {}

    def create(self, output_name: str) -> tuple[str, Path, Path]:
        self.cleanup_expired()
        token = uuid.uuid4().hex
        directory = self.root / token
        directory.mkdir()
        input_path = directory / "input.stl"
        step_path = directory / output_name
        self._jobs[token] = StoredConversion(directory, step_path, time.time())
        return token, input_path, step_path

    def get(self, token: str) -> StoredConversion | None:
        self.cleanup_expired()
        return self._jobs.get(token)

    def discard(self, token: str) -> None:
        job = self._jobs.pop(token, None)
        if job is not None:
            shutil.rmtree(job.directory, ignore_errors=True)

    def cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for token, job in list(self._jobs.items()):
            if job.created_at < cutoff:
                self.discard(token)

    def close(self) -> None:
        self._jobs.clear()
        shutil.rmtree(self.root, ignore_errors=True)


def _convert_with_preview(
    input_path: Path,
    step_path: Path,
    checkpoint_path: Path | None,
    segmentation_mode: SegmentationMode,
) -> tuple[ConversionResult, ShapePreview | None, str | None]:
    result = convert(
        input_path,
        step_path,
        checkpoint_path=checkpoint_path,
        segmentation_mode=segmentation_mode,
    )
    try:
        preview = result._comparison_preview or tessellate_shape(result._shape)
        deviations = preview_deviations(load_mesh(input_path).mesh, preview.positions)
        preview = replace(
            preview,
            deviations=deviations,
            deviation_scale=result.metrics.fidelity_tolerance,
        )
        return result, preview, None
    except Exception as exc:  # A preview failure must not invalidate a good STEP file.
        return result, None, str(exc)


def _safe_output_name(filename: str | None) -> str:
    stem = Path(filename or "model.stl").stem
    clean = "".join(character for character in stem if character.isalnum() or character in "-_")
    return f"{clean or 'model'}.step"


def _static_directory() -> Path:
    override = os.environ.get("MESH2CAD_WEB_STATIC")
    if override:
        return Path(override).resolve()
    return Path(str(resources.files("mesh2cad").joinpath("web_static")))


def create_app(
    *,
    store: ConversionStore | None = None,
    checkpoint_path: Path | None = None,
    segmentation_mode: SegmentationMode = "auto",
    dashboard_state_root: Path | None = None,
    dashboard_gpu_uuid: str | None = None,
):
    """Create the FastAPI application without binding a public network interface."""
    try:
        from fastapi import FastAPI, File, HTTPException, Request, UploadFile
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by minimal installs
        raise RuntimeError("Install the web dependencies with `pip install -e .[web]`") from exc

    conversion_store = store or ConversionStore()
    @asynccontextmanager
    async def lifespan(_app):
        yield
        conversion_store.close()

    app = FastAPI(
        title="mesh2cad local inference API",
        version="0.1.0",
        license_info={"name": "MIT", "identifier": "MIT"},
        description="Local STL-to-STEP reconstruction. Uploaded data is not transmitted.",
        lifespan=lifespan,
    )
    app.state.conversion_store = conversion_store
    app.state.checkpoint_path = checkpoint_path
    app.state.segmentation_mode = segmentation_mode
    state_root = dashboard_state_root or Path(
        os.environ.get("MESH2CAD_DASHBOARD_STATE", Path.home() / ".cache/mesh2cad/telemetry")
    )
    gpu_uuid = dashboard_gpu_uuid or os.environ.get("MESH2CAD_GPU_UUID")

    @app.middleware("http")
    async def disable_local_ui_cache(request, call_next):
        response = await call_next(request)
        if request.url.path in {
            "/",
            "/index.html",
            "/app.js",
            "/app.css",
            "/dashboard",
            "/dashboard.html",
            "/dashboard.js",
            "/dashboard.css",
        }:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "engine": segmentation_mode,
            "checkpoint": "loaded" if checkpoint_path is not None else "none",
        }

    @app.get("/api/dashboard")
    async def dashboard() -> dict[str, object]:
        return await collect_dashboard_snapshot(state_root, gpu_uuid=gpu_uuid)

    @app.get("/api/dashboard/stream")
    async def dashboard_stream(request: Request):
        async def events():
            while not await request.is_disconnected():
                snapshot = await collect_dashboard_snapshot(state_root, gpu_uuid=gpu_uuid)
                yield f"event: snapshot\ndata: {json.dumps(snapshot, separators=(',', ':'))}\n\n"
                await asyncio.sleep(2.0)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/convert")
    async def convert_upload(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        if not (file.filename or "").lower().endswith(".stl"):
            raise HTTPException(status_code=415, detail="Only STL files are supported.")

        output_name = _safe_output_name(file.filename)
        token, input_path, step_path = conversion_store.create(output_name)
        maximum = int(os.environ.get("MESH2CAD_MAX_UPLOAD_BYTES", "0"))
        received = 0
        try:
            with input_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    received += len(chunk)
                    if maximum > 0 and received > maximum:
                        raise HTTPException(
                            status_code=413, detail="The STL exceeds the configured limit."
                        )
                    destination.write(chunk)
            await file.close()
            result, preview, preview_error = await run_in_threadpool(
                _convert_with_preview,
                input_path,
                step_path,
                checkpoint_path,
                segmentation_mode,
            )
        except HTTPException:
            conversion_store.discard(token)
            raise
        except Exception as exc:
            conversion_store.discard(token)
            raise HTTPException(status_code=422, detail=f"Reconstruction failed: {exc}") from exc

        return {
            "token": token,
            "download_url": f"/api/download/{token}",
            "output_name": output_name,
            "mesh": {"vertices": result.mesh_vertices, "faces": result.mesh_faces},
            "mesh_quality": (
                None if result.mesh_quality is None else result.mesh_quality.to_dict()
            ),
            "surfaces": {
                "engine": result.segmentation_engine,
                "planes": len(result.segmentation.plane_regions),
                "cylinders": len(result.segmentation.cylinder_regions),
                "unassigned_faces": len(result.segmentation.unassigned_faces),
            },
            "metrics": result.metrics.to_dict(),
            "cad_quality": result.cad_quality(),
            "topology": None if result.topology is None else result.topology.to_dict(),
            "preview": None if preview is None else preview.to_dict(),
            "preview_error": preview_error,
        }

    @app.get("/api/download/{token}")
    def download(token: str):
        job = conversion_store.get(token)
        if job is None or not job.step_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="This download has expired or does not exist.",
            )
        return FileResponse(job.step_path, filename=job.step_path.name, media_type="model/step")

    static = _static_directory()
    if static.is_dir():
        @app.get("/dashboard")
        def dashboard_page():
            return FileResponse(static / "dashboard.html")

        app.mount("/", StaticFiles(directory=static, html=True), name="web")
    else:
        @app.get("/")
        def missing_frontend() -> dict[str, str]:
            return {"message": "Web assets are not built. Run scripts/build-web.ps1."}

    return app


def _available_port(preferred: int) -> int:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
        except OSError:
            probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _parser() -> argparse.ArgumentParser:
    checkpoint = default_checkpoint()
    parser = argparse.ArgumentParser(description="Launch the private local mesh2cad web app.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="fail instead of silently choosing another port when --port is unavailable",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoint,
        help="trained neural checkpoint, bundled model, or MESH2CAD_CHECKPOINT",
    )
    parser.add_argument(
        "--segmentation-engine",
        choices=("auto", "deterministic", "neural"),
        default=default_segmentation_engine(checkpoint),
        help="surface proposer mode (bundled reviewed models default to neural)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install web dependencies with `pip install -e .[web]`") from exc

    port = args.port if args.strict_port else _available_port(args.port)
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        threading.Timer(1.1, webbrowser.open, args=(url,)).start()
    print(f"mesh2cad local app: {url}")
    print("Files stay on this computer. Press Ctrl+C to stop.")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
    if args.segmentation_engine == "neural" and args.checkpoint is None:
        raise SystemExit("Neural segmentation engine requires --checkpoint")
    print(f"Segmentation engine: {args.segmentation_engine}")
    uvicorn.run(
        create_app(
            checkpoint_path=args.checkpoint,
            segmentation_mode=args.segmentation_engine,
        ),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
