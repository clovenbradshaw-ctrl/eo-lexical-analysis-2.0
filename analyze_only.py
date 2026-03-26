#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     EO LEXICAL ANALYSIS — Analysis Only (No Classification / Embedding)     ║
║                                                                              ║
║  Downloads pre-classified + pre-embedded data and runs geometric analysis.   ║
║  No API keys required. No classification. No embedding.                      ║
║                                                                              ║
║  Run:  python analyze_only.py                                                ║
║  Help: python analyze_only.py --help                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, subprocess, argparse, textwrap, re, shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bold(s):  return f"\033[1m{s}\033[0m"
def dim(s):   return f"\033[2m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def yellow(s):return f"\033[33m{s}\033[0m"
def red(s):   return f"\033[31m{s}\033[0m"
def cyan(s):  return f"\033[36m{s}\033[0m"
def header(title):
    w = 74
    print(f"\n{'─'*w}")
    print(f"  {bold(title)}")
    print(f"{'─'*w}")
def section(title):
    print(f"\n  {cyan('▸')} {bold(title)}")
def ok(msg):   print(f"  {green('✓')} {msg}")
def warn(msg): print(f"  {yellow('!')} {msg}")
def err(msg):  print(f"  {red('✗')} {msg}")
def info(msg): print(f"  {dim('·')} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "numpy":      "numpy",
    "scipy":      "scipy",
    "sklearn":    "scikit-learn",
    "matplotlib": "matplotlib",
    "tqdm":       "tqdm",
    "requests":   "requests",
}

def check_dependencies():
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing

def install_dependencies(packages):
    pip_names = [p for _, p in packages]
    print(f"  Installing: {', '.join(pip_names)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pip_names)


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE DOWNLOAD
#
# The embeddings.npz files are too large for GitHub and are stored on
# Google Drive. The repo contains placeholder files with the Drive URL.
# ─────────────────────────────────────────────────────────────────────────────

# Map of run directory -> {file: google_drive_url}
# These are the pre-computed datasets from the full pipeline.
RUN_DATASETS = {
    "run_2026-03-15_122636": {
        "embeddings_url": "https://drive.google.com/file/d/1CP7d5ZNa3p4HR-TKt1dz737yQixE-Zct/view?usp=sharing",
        "description": "Full corpus: UD + FLORES + arXiv + Bible + Philosophy (19k+ clauses, 41 languages)",
    },
    "run_2026-03-19_144302": {
        "embeddings_url": "https://drive.google.com/file/d/1x9RV3ZHWquWlvnjumR2V0QsrRFaDVF95/view?usp=sharing",
        "description": "Second run: UD + arXiv + Bible + Philosophy (3.5k+ clauses, English-focused)",
    },
}


def extract_gdrive_file_id(url: str) -> str:
    """Extract the file ID from various Google Drive URL formats."""
    # https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    m = re.search(r'/file/d/([^/]+)', url)
    if m:
        return m.group(1)
    # https://drive.google.com/open?id=FILE_ID
    m = re.search(r'[?&]id=([^&]+)', url)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract file ID from URL: {url}")


def download_from_gdrive(file_id: str, dest_path: Path, desc: str = "file"):
    """Download a file from Google Drive, handling the large-file confirmation page."""
    import requests

    info(f"Downloading {desc} from Google Drive...")
    info(f"File ID: {file_id}")

    session = requests.Session()

    try:
        # Initial request
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True, timeout=30)
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Google Drive: {e}\n\n"
            f"  Please download the file manually:\n"
            f"    https://drive.google.com/file/d/{file_id}/view?usp=sharing\n\n"
            f"  Save it to: {dest_path}\n"
        )

    # Check for virus scan warning / confirmation token
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break

    if confirm_token:
        url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
        response = session.get(url, stream=True)

    # If still getting HTML (large file warning), try the confirm=t approach
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        response = session.get(url, stream=True)

    # Write to disk
    total = int(response.headers.get('content-length', 0))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix('.tmp')

    try:
        from tqdm import tqdm as _tqdm
        with open(tmp_path, 'wb') as f:
            with _tqdm(total=total, unit='B', unit_scale=True, desc=f"  {desc}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except ImportError:
        with open(tmp_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {desc}: {downloaded:,} / {total:,} bytes ({pct:.0f}%)", end="", flush=True)
            print()

    # Verify it's actually a numpy file and not an HTML error page
    if tmp_path.stat().st_size < 1000:
        content = tmp_path.read_bytes()
        if b'<html' in content.lower() or b'<!doctype' in content.lower():
            tmp_path.unlink()
            raise RuntimeError(
                f"Google Drive returned an HTML page instead of the file. "
                f"The file may require manual download.\n"
                f"Please download manually from:\n"
                f"  https://drive.google.com/file/d/{file_id}/view?usp=sharing\n"
                f"and save it to: {dest_path}"
            )

    tmp_path.rename(dest_path)
    ok(f"Downloaded {desc}: {dest_path} ({dest_path.stat().st_size:,} bytes)")


def ensure_embeddings(run_dir: Path) -> Path:
    """
    Ensure embeddings.npz exists and is a valid numpy file.
    If it's a placeholder (contains a URL), download the real file from Google Drive.
    """
    emb_path = run_dir / "embeddings.npz"

    if not emb_path.exists():
        err(f"embeddings.npz not found in {run_dir}")
        sys.exit(1)

    # Check if it's a placeholder (small file containing a URL)
    if emb_path.stat().st_size < 500:
        content = emb_path.read_text(errors='replace').strip()
        if 'drive.google.com' in content:
            info(f"embeddings.npz is a placeholder pointing to Google Drive")
            file_id = extract_gdrive_file_id(content)
            # Back up placeholder
            placeholder_path = emb_path.with_suffix('.npz.url')
            if not placeholder_path.exists():
                shutil.copy2(emb_path, placeholder_path)
            try:
                download_from_gdrive(file_id, emb_path, desc="embeddings.npz")
            except RuntimeError as e:
                err(str(e))
                print()
                err("Automatic download failed. Please download manually:")
                print(f"\n    URL:  {content}")
                print(f"    Save to: {emb_path}\n")
                sys.exit(1)
        else:
            err(f"embeddings.npz exists but is only {emb_path.stat().st_size} bytes and doesn't contain a URL")
            sys.exit(1)

    # Validate it's a real numpy file
    try:
        import numpy as np
        data = np.load(emb_path, allow_pickle=True)
        required_keys = {"vectors", "q1", "q2", "q3", "ids", "language"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            err(f"embeddings.npz is missing required keys: {missing_keys}")
            sys.exit(1)
        ok(f"embeddings.npz validated: {data['vectors'].shape[0]:,} vectors × {data['vectors'].shape[1]} dims")
        data.close()
    except Exception as e:
        if "pickle" in str(e).lower() or "Failed to interpret" in str(e):
            # Corrupted or placeholder — try to recover from URL
            err(f"embeddings.npz is not a valid numpy file: {e}")
            # Check if we have a .url backup
            url_path = emb_path.with_suffix('.npz.url')
            if url_path.exists():
                content = url_path.read_text().strip()
                if 'drive.google.com' in content:
                    warn("Attempting re-download from saved URL...")
                    file_id = extract_gdrive_file_id(content)
                    download_from_gdrive(file_id, emb_path, desc="embeddings.npz")
                    return ensure_embeddings(run_dir)  # recursive validate
            # Check the RUN_DATASETS table
            run_name = run_dir.name
            if run_name in RUN_DATASETS:
                url = RUN_DATASETS[run_name]["embeddings_url"]
                file_id = extract_gdrive_file_id(url)
                warn(f"Attempting download from known URL for {run_name}...")
                download_from_gdrive(file_id, emb_path, desc="embeddings.npz")
                return ensure_embeddings(run_dir)
            sys.exit(1)
        raise

    return emb_path


def ensure_classified(run_dir: Path) -> Path:
    """Ensure classified.jsonl exists in the run directory."""
    cls_path = run_dir / "classified.jsonl"
    if not cls_path.exists():
        err(f"classified.jsonl not found in {run_dir}")
        err("This file should be in the repository already.")
        sys.exit(1)
    # Quick validation
    with open(cls_path) as f:
        first_line = f.readline()
    try:
        rec = json.loads(first_line)
        if "classifications" not in rec:
            warn("classified.jsonl doesn't look like a classification file (no 'classifications' key)")
    except json.JSONDecodeError:
        err("classified.jsonl is not valid JSONL")
        sys.exit(1)

    n_lines = sum(1 for _ in open(cls_path))
    ok(f"classified.jsonl: {n_lines:,} classified clauses")
    return cls_path


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABLE RUN SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_available_runs() -> list:
    """Find all run_* directories that have the required files."""
    runs = []
    for d in sorted(Path(".").glob("run_*")):
        if d.is_dir() and (d / "classified.jsonl").exists():
            has_emb = (d / "embeddings.npz").exists()
            runs.append({"dir": d, "has_embeddings": has_emb})
    return runs


def select_run() -> Path:
    """Interactive run directory selection."""
    runs = find_available_runs()

    if not runs:
        err("No run directories found with classified.jsonl")
        err("Run the full pipeline (app2.py) first to generate data,")
        err("or ensure run directories are present in the current directory.")
        sys.exit(1)

    if len(runs) == 1:
        run_dir = runs[0]["dir"]
        ok(f"Found one dataset: {run_dir}")
        return run_dir

    print("\n  Available datasets:\n")
    for i, run in enumerate(runs, 1):
        d = run["dir"]
        name = d.name
        desc = RUN_DATASETS.get(name, {}).get("description", "")
        emb_status = green("✓ embeddings") if run["has_embeddings"] else yellow("⬇ needs download")
        cls_size = (d / "classified.jsonl").stat().st_size
        print(f"    {bold(str(i))}. {name}")
        if desc:
            print(f"       {desc}")
        print(f"       classified.jsonl: {cls_size:,} bytes  |  {emb_status}")
        print()

    while True:
        try:
            choice = input(f"  {bold('Select dataset')} [1-{len(runs)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]["dir"]
        except (ValueError, EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        print(f"  Please enter a number between 1 and {len(runs)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EO Lexical Analysis — Analysis Only (no classification or embedding)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
          This script downloads pre-classified and pre-embedded data and runs
          ONLY the geometric analysis (Phase 4 + Phase 5). No API keys needed.

          Examples:
            python analyze_only.py                           # interactive run selection
            python analyze_only.py --run-dir run_2026-03-15_122636
            python analyze_only.py --force                   # recompute everything
            python analyze_only.py --list                    # show available datasets
        """)
    )
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to a specific run directory")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation of all analysis (ignore cached results.json)")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    header("EO LEXICAL ANALYSIS — Analysis Only")
    print("""
  This tool runs geometric analysis on pre-classified, pre-embedded data.
  No API keys. No classification. No embedding. Just analysis.

  It will:
    1. Download embeddings from Google Drive (if needed)
    2. Run Phase 4: z-scores, proportionality, ARI, helix tests, etc.
    3. Run Phase 5: centroid classifier, composite tests, exemplars
    """)

    # ── Check dependencies ────────────────────────────────────────────────────
    section("Checking dependencies")
    missing = check_dependencies()
    if missing:
        warn(f"Missing packages: {', '.join(p for _,p in missing)}")
        try:
            ans = input(f"\n  {bold('Install them now?')} (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not ans or ans.startswith('y'):
            install_dependencies(missing)
            ok("Installed. Re-importing...")
            import importlib
            for pkg, _ in missing:
                try: importlib.import_module(pkg)
                except: pass
        else:
            err("Cannot continue without required packages.")
            sys.exit(1)
    else:
        ok("All dependencies satisfied")

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        runs = find_available_runs()
        if not runs:
            info("No run directories found")
        for run in runs:
            d = run["dir"]
            desc = RUN_DATASETS.get(d.name, {}).get("description", "")
            print(f"  {d.name}  {desc}")
        sys.exit(0)

    # ── Select run directory ──────────────────────────────────────────────────
    section("Selecting dataset")
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            err(f"Directory not found: {run_dir}")
            sys.exit(1)
        ok(f"Using: {run_dir}")
    else:
        run_dir = select_run()

    # ── Download / validate data ──────────────────────────────────────────────
    header("DATA PREPARATION")
    section("Checking classified.jsonl")
    classified_file = ensure_classified(run_dir)

    section("Checking embeddings.npz")
    embeddings_file = ensure_embeddings(run_dir)

    # ── Import analysis functions from app2.py ────────────────────────────────
    # We import app2 as a module to reuse all 5000+ lines of analysis code
    # without duplication.
    section("Loading analysis engine")
    app2_path = Path(__file__).parent / "app2.py"
    if not app2_path.exists():
        err("app2.py not found — it must be in the same directory as this script.")
        err("The analysis functions live in app2.py.")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location("app2", str(app2_path))
    app2 = importlib.util.module_from_spec(spec)

    # Suppress the interactive setup and corpus download code during import
    # by temporarily replacing sys.argv
    old_argv = sys.argv
    sys.argv = ["app2.py", "--help-hidden"]  # won't trigger main()
    try:
        spec.loader.exec_module(app2)
    except SystemExit:
        pass  # argparse --help would exit
    finally:
        sys.argv = old_argv

    ok("Analysis engine loaded from app2.py")

    # ── Detect which models were used ─────────────────────────────────────────
    models_used = []
    with open(classified_file) as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            try:
                rec = json.loads(line)
                cls = rec.get("classifications", {})
                for m in cls:
                    if m == "claude" and "claude" not in models_used:
                        models_used.append("claude")
                    if m in ("gpt4", "gpt-4o", "gpt-4o-mini") and "gpt4" not in models_used:
                        models_used.append("gpt4")
            except Exception:
                pass
    if not models_used:
        models_used = ["claude"]
    info(f"Models detected in data: {', '.join(models_used)}")

    # ── Phase 4: Analysis ─────────────────────────────────────────────────────
    header("PHASE 4 — ANALYSIS (UNIFIED POOL)")
    print("""
  All clauses combined into a single unified analysis pool.
  Metrics computed on all clauses with best-available labels:

  (a) Per-axis z-score — geometric separation vs random baseline
  (b) Proportionality — distance scales with axis-difference count
  (c) Axis independence (ARI) — are the three axes orthogonal?
  (d) Operator/face z-scores — combinatorial structure
  (e) Coordinate geometry — axis spacing patterns
  (f) Helix dependency — Mode-Domain correlation structure
    """)

    results = app2.run_analysis(
        embeddings_file=embeddings_file,
        classified_file=classified_file,
        run_dir=run_dir,
        models_used=models_used,
        force=args.force,
    )

    # ── Phase 5: Centroids ────────────────────────────────────────────────────
    header("PHASE 5 — CENTROID CLASSIFIER")
    print("""
  Computes geometric centroids for each EO cell from consensus-classified
  clauses, then classifies ALL clauses by nearest centroid — no AI
  classifiers involved, pure geometry.

  Tests whether the 27 cells are stable geometric regions that can
  address new text without any classification prompt.
    """)

    centroid_results, centroid_file = app2.run_centroids(
        embeddings_file=embeddings_file,
        classified_file=classified_file,
        run_dir=run_dir,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    header("COMPLETE")
    print(f"""
  Output directory: {run_dir}/

  Key files:
    analysis_report.txt    Full results with plain-English commentary
    results.json           Raw numbers (for further analysis)
    centroids.npz          27-cell centroid vectors
    centroid_results.json  Centroid classifier accuracy
    composite_test.json    Geometry tests (reconstruction, cross-axis)
    helix_geometry.json    Helix dependency tests
    coord_geometry.json    Coordinate metric analysis
    exemplars.json         Top-100 exemplars per cell
    figures/               PCA/LDA projections, z-score charts

  No API keys were used. No classifications were performed.
  No embeddings were generated. All analysis was done on pre-computed data.
    """)


if __name__ == "__main__":
    main()
