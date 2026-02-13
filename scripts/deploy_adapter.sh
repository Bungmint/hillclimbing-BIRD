#!/usr/bin/env bash
# Deploy a Tinker LoRA adapter to the Modal volume for vLLM inference.
#
# Usage:
#   # From a tinker:// sampler path directly:
#   ./scripts/deploy_adapter.sh "tinker://9abf1dc5-0b87-5c3f-922c-3d89a488bb7f:train:0/sampler_weights/000010" grpo_step10
#
#   # From a checkpoints.jsonl file (uses the last entry's sampler_path):
#   ./scripts/deploy_adapter.sh outputs/tinker_grpo/20260212_031638_Qwen-Qwen3-8B/checkpoints.jsonl grpo_step10
#
# Then run inference:
#   modal run modal_app.py --model-preset qwen-8b --lora-adapter-path /adapters/grpo_step10 --limit 200

set -euo pipefail

usage() {
  echo "Usage: $0 <tinker_sampler_path_or_checkpoints_jsonl> <adapter_name> [--no-upload]"
  echo ""
  echo "Arguments:"
  echo "  source         A tinker:// sampler URI, or path to a checkpoints.jsonl file"
  echo "  adapter_name   Name for the adapter on the Modal volume (e.g. grpo_step10)"
  echo "  --no-upload    Download only to local /tmp (debug path), skip Modal upload"
  exit 1
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command '$1'."
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOTENV_PATH="${REPO_ROOT}/.env"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
fi

SOURCE="$1"
ADAPTER_NAME="$2"
NO_UPLOAD=false
if [[ "${3:-}" == "--no-upload" ]]; then
    NO_UPLOAD=true
elif [[ "${3:-}" != "" ]]; then
    usage
fi

MODAL_VOLUME="bird-rl-adapters"
DOWNLOAD_DIR="/tmp/tinker_adapters/${ADAPTER_NAME}"
MODAL_DEPLOY_SCRIPT="${SCRIPT_DIR}/modal_deploy_adapter.py"

require_command python3

if [[ "$NO_UPLOAD" == false ]]; then
  require_command modal
fi

if [[ -z "${TINKER_API_KEY:-}" && -f "${DOTENV_PATH}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${DOTENV_PATH}"
  set +a
  echo "Loaded environment variables from ${DOTENV_PATH}"
fi

if [[ "$NO_UPLOAD" == true && -z "${TINKER_API_KEY:-}" ]]; then
  fail "TINKER_API_KEY is required for --no-upload local download mode."
fi

# ---------------------------------------------------------------------------
# Resolve the sampler path
# ---------------------------------------------------------------------------
if [[ "$SOURCE" == tinker://* ]]; then
  SAMPLER_PATH="$SOURCE"
elif [[ -f "$SOURCE" ]]; then
  # Parse the last line of checkpoints.jsonl for sampler_path.
  SAMPLER_PATH="$(
    python3 - "$SOURCE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    print("ERROR: checkpoints.jsonl is empty", file=sys.stderr)
    raise SystemExit(1)
try:
    last = json.loads(rows[-1])
except json.JSONDecodeError as exc:
    print(f"ERROR: invalid JSON in last line of {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

sampler_path = last.get("sampler_path")
if not isinstance(sampler_path, str) or not sampler_path:
    print("ERROR: last checkpoint has no sampler_path", file=sys.stderr)
    raise SystemExit(1)
print(sampler_path)
PY
)"
  echo "Resolved sampler_path from checkpoints.jsonl: ${SAMPLER_PATH}"
else
  fail "'$SOURCE' is not a tinker:// URI or a valid file path."
fi

# ---------------------------------------------------------------------------
# Direct Modal-side upload (no local adapter download)
# ---------------------------------------------------------------------------
if [[ "$NO_UPLOAD" == false ]]; then
  if [[ ! -f "${MODAL_DEPLOY_SCRIPT}" ]]; then
    fail "Modal deploy script not found: ${MODAL_DEPLOY_SCRIPT}"
  fi
  if [[ -z "${TINKER_API_KEY:-}" ]]; then
    echo "TINKER_API_KEY not set locally. Falling back to Modal secret 'tinker-credentials'."
  fi
  echo ""
  echo "==> Uploading adapter directly on Modal volume '${MODAL_VOLUME}' (no local artifact download) ..."
  modal run "${MODAL_DEPLOY_SCRIPT}" \
    --sampler-path "${SAMPLER_PATH}" \
    --adapter-name "${ADAPTER_NAME}"

  echo ""
  echo "==> Done! Run inference with:"
  echo "  modal run modal_app.py --model-preset qwen-8b --lora-adapter-path /adapters/${ADAPTER_NAME} --limit 200"
  exit 0
fi

echo ""
echo "==> Downloading adapter locally to ${DOWNLOAD_DIR} (--no-upload mode) ..."
rm -rf "${DOWNLOAD_DIR}"
mkdir -p "${DOWNLOAD_DIR}"

python3 - "$SAMPLER_PATH" "$DOWNLOAD_DIR" <<'PY'
import os
import pathlib
import tarfile
import urllib.request
import sys

try:
    import tinker
except ModuleNotFoundError:
    print("ERROR: Python package 'tinker' is not installed.", file=sys.stderr)
    print("Install with: uv sync --extra rl", file=sys.stderr)
    raise SystemExit(1)

sampler_path = sys.argv[1]
download_dir = pathlib.Path(sys.argv[2])
archive_path = download_dir / "archive.tar"

api_key = os.getenv("TINKER_API_KEY")
if not api_key:
    print("ERROR: TINKER_API_KEY is not set.", file=sys.stderr)
    raise SystemExit(1)

service_client = tinker.ServiceClient(api_key=api_key)
rest_client = service_client.create_rest_client()
archive_url = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()

print(f"Downloading from signed URL (expires {archive_url.expires}) ...")
urllib.request.urlretrieve(archive_url.url, archive_path)
print("Extracting...")
with tarfile.open(archive_path) as tar:
    tar.extractall(download_dir)
archive_path.unlink(missing_ok=True)
print("Download complete.")
PY

echo ""
echo "==> Download-only complete (--no-upload). Adapter is at: ${DOWNLOAD_DIR}"
