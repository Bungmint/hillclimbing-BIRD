"""Deploy a Tinker adapter directly into a Modal volume.

This avoids downloading adapter artifacts through local disk/network.

Examples:
  modal run scripts/modal_deploy_adapter.py \
    --sampler-path "tinker://.../sampler_weights/000010" \
    --adapter-name grpo_step10
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import modal

APP_NAME = "bird-adapter-deployer"
ADAPTER_VOLUME_NAME = "bird-rl-adapters"
TINKER_SECRET_NAME = "tinker-credentials"

app = modal.App(APP_NAME)

# If TINKER_API_KEY is set locally, inject it directly for this run.
# Otherwise, fall back to an existing Modal secret.
_local_tinker_api_key = os.getenv("TINKER_API_KEY")
if _local_tinker_api_key:
    tinker_secret = modal.Secret.from_dict({"TINKER_API_KEY": _local_tinker_api_key})
else:
    tinker_secret = modal.Secret.from_name(TINKER_SECRET_NAME, required_keys=["TINKER_API_KEY"])

image = modal.Image.debian_slim(python_version="3.11").pip_install("tinker>=0.6.1")
adapter_volume = modal.Volume.from_name(ADAPTER_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
    secrets=[tinker_secret],
    volumes={"/adapters": adapter_volume},
)
def deploy_adapter_remote(
    *,
    sampler_path: str,
    adapter_name: str,
    overwrite: bool = True,
) -> dict[str, object]:
    import tinker

    if not sampler_path.startswith("tinker://"):
        raise ValueError(f"Expected tinker:// sampler path, got: {sampler_path}")

    target_dir = Path("/adapters") / adapter_name
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Adapter path already exists: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = Path("/tmp") / f"{adapter_name}.tar"

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not set in Modal runtime.")

    service_client = tinker.ServiceClient(api_key=api_key)
    rest_client = service_client.create_rest_client()
    archive_url = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()

    urllib.request.urlretrieve(archive_url.url, archive_path)
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)
    archive_path.unlink(missing_ok=True)

    adapter_volume.commit()

    file_count = sum(1 for path in target_dir.rglob("*") if path.is_file())
    return {
        "volume": ADAPTER_VOLUME_NAME,
        "adapter_name": adapter_name,
        "target_path": str(target_dir),
        "file_count": file_count,
        "source_sampler_path": sampler_path,
    }


@app.local_entrypoint()
def main(
    sampler_path: str,
    adapter_name: str,
    overwrite: bool = True,
) -> None:
    result = deploy_adapter_remote.remote(
        sampler_path=sampler_path,
        adapter_name=adapter_name,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2))
