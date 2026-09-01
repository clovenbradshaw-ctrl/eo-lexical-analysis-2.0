"""Registry of local CPU models used as the 3rd/4th classifiers.

Both are 7B-class instruct models, Q4_K_M GGUF, chosen for:
  - a different lab/training lineage each (Alibaba, Mistral AI) and from
    Claude/GPT-4o-mini, which is the whole point: WHY-THESE-THREE.md's
    "poverty of stimulus" section notes Claude and GPT-4 are not
    independent observers (shared pretraining). Two more, differently
    -trained models sharpen that estimate; they still don't make it a
    human-annotator study.
  - single-file GGUF availability (no multi-part reassembly).
  - fitting comfortably in this box's 4 vCPU / 15GB RAM.

Fetch with ./download_models.sh before use; weights/*.gguf is gitignored
(multi-GB binaries don't belong in the repo).
"""
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODELS = {
    "qwen": dict(
        label="Qwen2.5-7B-Instruct (Q4_K_M)",
        path=WEIGHTS_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf",
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        hf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    "mistral": dict(
        label="Mistral-7B-Instruct-v0.3 (Q4_K_M)",
        path=WEIGHTS_DIR / "mistral-7b-instruct-v0.3-q4_k_m.gguf",
        hf_repo="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
        hf_file="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    ),
}


def check_available():
    missing = [k for k, v in MODELS.items() if not v["path"].exists()]
    if missing:
        raise FileNotFoundError(
            f"missing local model weights for {missing} — run "
            f"alt_structures/local_models/download_models.sh first"
        )
