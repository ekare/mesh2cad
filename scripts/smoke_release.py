"""Exercise a running release executable through its public local API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = f"mesh2cad-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="demo.stl"\r\n'
        "Content-Type: model/stl\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("stl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-engine", choices=("auto", "deterministic", "neural"))
    parser.add_argument("--expect-checkpoint", choices=("none", "loaded"))
    args = parser.parse_args()

    health_url = args.base_url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + 45
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                health = json.load(response)
                if health["status"] == "ok":
                    break
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("Packaged web server did not become healthy") from exc
            time.sleep(0.5)
    if args.expect_engine is not None and health.get("engine") != args.expect_engine:
        raise RuntimeError(f"Unexpected packaged engine: {health}")
    if args.expect_checkpoint is not None and health.get("checkpoint") != args.expect_checkpoint:
        raise RuntimeError(f"Unexpected packaged checkpoint state: {health}")

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/dashboard", timeout=10) as response:
        dashboard = response.read()
    if b"Training Observatory" not in dashboard:
        raise RuntimeError("Packaged dashboard asset is missing")

    body, boundary = _multipart(args.stl)
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/api/convert",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not payload["topology"]["valid"] or payload["surfaces"]["cylinders"] != 1:
        raise RuntimeError(f"Unexpected conversion response: {payload}")
    if payload["mesh_quality"]["repair_recommended"] or not payload["mesh_quality"]["watertight"]:
        raise RuntimeError(f"Unexpected mesh-quality response: {payload['mesh_quality']}")
    with urllib.request.urlopen(args.base_url + payload["download_url"], timeout=30) as response:
        args.output.write_bytes(response.read())
    if b"ISO-10303-21" not in args.output.read_bytes()[:100]:
        raise RuntimeError("Downloaded file is not a STEP exchange file")
    print("Packaged web upload/conversion/download smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
