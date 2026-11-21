#!/usr/bin/env python3
"""Read a Vesuvius Challenge segment with Reader V3 on a CPU-only machine, in one command.

    python read_segment_cpu.py <surface-volume.zarr URL or path> [--roi Y0 Y1 X0 X1] [--out out/name]

What it does
  1. Downloads the Reader V3 checkpoint (Domenico Russo, huggingface.co/domenicor046/reader-v3) once and checks its sha256.
  2. Runs villa's flat ink inference (`vesuvius.ink_detection.inference.infer`) on the CPU, streaming only the chunks
     it needs from the public open-data bucket (or reading a local Zarr).
  3. Writes the 8-bit ink TIFF(s) and a contrast-stretched PNG preview (median -> 99.5th percentile, as in Reader V3's README).

Needs: CPU torch, villa's `vesuvius` package on PYTHONPATH (or installed with --no-deps), and either the native
`vc_reader` or the pure-Python `vc-reader-lite` from this repo. See README.md.
"""
from __future__ import annotations

import argparse, hashlib, os, sys, time, urllib.request
from pathlib import Path

CKPT_URL = "https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth"
CKPT_SHA256 = "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a"


def fetch_checkpoint(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading Reader V3 checkpoint -> {dest}", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(CKPT_URL, tmp)
        tmp.replace(dest)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != CKPT_SHA256:
        sys.exit(f"checkpoint sha256 mismatch for {dest}; delete it and retry")
    return dest


def auto_threads() -> int:
    """Threads = cores that are idle right now. Oversubscribing a busy CPU made PyTorch 5-8x slower in our tests."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    try:  # Linux: sample /proc/stat for half a second
        def snap():
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            return v[3] + v[4], sum(v)  # idle + iowait, total
        i0, t0 = snap(); time.sleep(0.5); i1, t1 = snap()
        idle = cpus * (i1 - i0) / max(1, t1 - t0)
        return max(1, min(cpus, int(idle + 0.25)))
    except OSError:  # Windows/macOS: physical cores is the usual sweet spot; leave one for the OS
        return max(1, (os.cpu_count() or 2) // 2)


def preview_png(tif: Path, png: Path) -> None:
    import numpy as np, tifffile
    from PIL import Image
    a = tifffile.imread(str(tif)).astype("float32")
    valid = a[a > 0]
    lo, hi = (float(np.median(valid)), float(np.percentile(valid, 99.5))) if valid.size else (0.0, 255.0)
    Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")).save(png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surface_volume", help="surface-volume Zarr: s3://..., https://... or a local path")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"), help="pixel rectangle (half-open); default: whole segment")
    ap.add_argument("--out", default="out/ink", help="output prefix (default out/ink -> out/ink.tif, out/ink.png)")
    ap.add_argument("--direction", choices=("forward", "reverse", "both"), default="forward",
                    help="layer order; try 'both' once per segment and keep the one that shows letters")
    ap.add_argument("--overlap", type=float, default=0.5, help="0.5 = Reader V3 default; 0.25 is ~1.5x faster, slightly blockier")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 = auto (idle cores). More threads than idle cores is MUCH slower")
    ap.add_argument("--batch-size", type=int, default=2, help="patches per forward pass (2-4 is a little faster on CPU)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (~30 s warm-up, ~1.25x faster after; exact)")
    ap.add_argument("--bf16", action="store_true",
                    help="EXPERIMENTAL: bfloat16 autocast on CPU (~1.7x faster on CPUs with AMX/AVX512-BF16; "
                         "small output differences, see README). villa only autocasts on CUDA, so this patches that one call.")
    ap.add_argument("--checkpoint", default=str(Path.home() / ".cache" / "reader-v3" / "reader-v3.pth"))
    ap.add_argument("--villa", default=os.environ.get("VILLA_DIR"), help="path to a villa checkout (adds villa/vesuvius/src to sys.path)")
    a = ap.parse_args()

    if a.villa:
        sys.path.insert(0, str(Path(a.villa) / "vesuvius" / "src"))
    import torch
    torch.set_num_threads(a.threads or auto_threads())
    try:
        import vc_reader
        backend = "vc-reader-lite (pure Python)" if getattr(vc_reader, "LITE", False) else "native vc_reader"
    except ImportError:
        sys.exit("vc_reader is missing: `pip install ./vcr_lite` (no compiler) or build villa/vc_reader (see README)")
    from vesuvius.ink_detection.inference import infer
    if a.bf16:
        # villa's flat loop uses `torch.autocast("cuda") if cuda else nullcontext()`; give CPU a bf16 autocast instead.
        infer.nullcontext = lambda: torch.autocast("cpu", dtype=torch.bfloat16)

    ckpt = fetch_checkpoint(Path(a.checkpoint))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    argv = [a.surface_volume, str(ckpt), str(out.with_suffix(".tif")), "--device", "cpu",
            "--overlap", str(a.overlap), "--blend-mode", "hann", "--direction", a.direction,
            "--batch-size", str(a.batch_size)] + ([] if a.compile else ["--no-compile"])
    if a.roi:
        argv += ["--roi", *map(str, a.roi)]
    print(f"backend: {backend}; torch {torch.__version__}, {torch.get_num_threads()} threads, "
          f"batch {a.batch_size}, compile={a.compile}, bf16={a.bf16}", flush=True)
    t0 = time.time()
    infer.main(argv)
    secs = time.time() - t0
    tifs = [out.with_suffix(".tif")] if a.direction != "both" else [out.with_suffix(".tif"), out.with_name(out.name + "_reverse.tif")]
    for tif in tifs:
        if tif.exists():
            preview_png(tif, tif.with_suffix(".png"))
            print(f"wrote {tif} and {tif.with_suffix('.png')}")
    print(f"inference took {secs:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
