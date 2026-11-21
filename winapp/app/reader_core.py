#!/usr/bin/env python3
"""Vesuvius Reader (desktop app) - the part that does the actual work.

Runs in its own process so the window stays responsive and "Stop" can simply end it.
It prints one JSON object per line, prefixed with "@@ ", which the app turns into the
progress bar and status text. Anything else it prints is ordinary log text.

    python reader_core.py run   <surface-volume> --out <prefix> [--roi Y0 Y1 X0 X1] [...]
    python reader_core.py preview <surface-volume> --out preview.png   (fast low-res look)
    python reader_core.py selftest                                     (tiny end-to-end check)

Based on read_segment_cpu.py from github.com/jamiejmann/vesuvius-cpu-reader (MIT).
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT_URL = "https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth"
CKPT_SHA256 = "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a"
SELFTEST_VOLUME = ("s3://vesuvius-challenge-open-data/PHerc1447/segments/20260928183143-w058/"
                   "surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr")
SELFTEST_ROI = (2200, 2584, 2200, 2968)  # 384 x 768 px, a few patches, inside written area


def emit(**kw) -> None:
    print("@@ " + json.dumps(kw), flush=True)


def data_dir() -> Path:
    """Per-user folder for the model and caches (%LOCALAPPDATA%\\VesuviusReader on Windows)."""
    base = os.environ.get("VESUVIUS_READER_HOME")
    if base:
        p = Path(base)
    elif os.name == "nt":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "VesuviusReader"
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vesuvius-reader"
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_paths() -> None:
    """Find villa's `vesuvius` package and vc_reader(-lite) next to the app (bundle layout) or via env."""
    cands = [os.environ.get("VILLA_DIR"), HERE.parent / "villa", HERE / "villa"]
    for c in cands:
        if c and (Path(c) / "vesuvius" / "src" / "vesuvius").is_dir():
            sys.path.insert(0, str(Path(c) / "vesuvius" / "src"))
            break
    lite = HERE.parent / "vcr_lite" / "src"
    if lite.is_dir():
        sys.path.append(str(lite))


# ---------------------------------------------------------------- model download
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_checkpoint(dest: Path | None = None) -> Path:
    dest = dest or data_dir() / "reader-v3.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == CKPT_SHA256:
        return dest
    emit(stage="download", msg="Downloading the Reader V3 model (about 140 MB, first time only)...", frac=0.0)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(CKPT_URL, headers={"User-Agent": "vesuvius-reader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0.0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf); got += len(buf)
            if total and time.time() - last > 0.5:
                last = time.time()
                emit(stage="download", msg=f"Downloading the model: {got/1e6:.0f} of {total/1e6:.0f} MB", frac=got / total)
    if sha256_of(tmp) != CKPT_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The downloaded model file looks damaged (checksum mismatch). Please try again.")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------- machine sizing
def cpu_info() -> dict:
    logical = os.cpu_count() or 2
    physical = None
    idle = None
    avail_gb = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        busy = psutil.cpu_percent(interval=0.5)
        idle = logical * (1 - busy / 100.0)
        avail_gb = psutil.virtual_memory().available / 1024**3
    except Exception:
        pass
    if physical is None:
        physical = max(1, logical // 2) if logical > 2 else logical
    return {"logical": logical, "physical": physical, "idle": idle, "avail_gb": avail_gb}


def auto_threads(info: dict | None = None) -> int:
    """Physical cores, but never more than are idle right now.

    Oversubscribing a CPU that is busy with other things made PyTorch 5-8x slower in our tests
    (13x end to end), so it is better to use fewer threads than too many."""
    info = info or cpu_info()
    n = info["physical"]
    if info["idle"] is not None:
        n = min(n, int(info["idle"] + 0.25))
    return max(1, n)


def auto_batch(info: dict | None = None) -> int:
    info = info or cpu_info()
    return 1 if (info["avail_gb"] is not None and info["avail_gb"] < 3.0) else 2


# ---------------------------------------------------------------- outputs
def preview_png(tif: Path, png: Path) -> None:
    import numpy as np, tifffile
    from PIL import Image
    a = tifffile.imread(str(tif)).astype("float32")
    valid = a[a > 0]
    lo, hi = (float(np.median(valid)), float(np.percentile(valid, 99.5))) if valid.size else (0.0, 255.0)
    Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")).save(png)


def segment_preview(volume: str, out_png: Path, max_side: int = 1400) -> dict:
    """A quick low-resolution look at the segment (middle layer of a coarse pyramid level)."""
    import numpy as np
    from PIL import Image
    from vc_reader.reader import open_node
    root = open_node(volume) if "://" in volume else open_node(str(Path(volume)))
    full = root["0"].shape if not hasattr(root, "shape") else root.shape
    if hasattr(root, "shape"):
        arr, level = root, 0
    else:
        arr, level = None, 0
        for lv in range(6, -1, -1):
            try:
                a = root[str(lv)]
            except Exception:
                continue
            if max(a.shape[1:]) >= 600 or lv == 0:
                arr, level = a, lv
                break
        arr = arr if arr is not None else root["0"]
    z = arr.shape[0] // 2
    im = np.asarray(arr[z]).astype("float32")
    v = im[im > 0]
    lo, hi = (float(np.percentile(v, 1)), float(np.percentile(v, 99.5))) if v.size else (0.0, 255.0)
    img = Image.fromarray((np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8"))
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return {"full_h": int(full[1]), "full_w": int(full[2]), "layers": int(full[0]), "level": level}


# ---------------------------------------------------------------- the run
class _Progress:
    """Wraps villa's patch loader so we can count finished patches and estimate time left."""

    def __init__(self, loader, phase: str, total_patches: int | None):
        self.loader, self.phase, self.total = loader, phase, total_patches
        try:
            self.total = self.total or len(loader.dataset)
        except Exception:
            pass

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        done, t0, last = 0, time.time(), 0.0
        emit(stage="infer", phase=self.phase, done=0, total=self.total, frac=0.0,
             msg="Reading the scroll... (the first patches are the slowest)")
        for batch in self.loader:
            yield batch
            try:
                done += int(batch[1].shape[0])
            except Exception:
                done += 1
            now = time.time()
            if now - last > 1.0 or (self.total and done >= self.total):
                last = now
                frac = done / self.total if self.total else None
                eta = (now - t0) / done * (self.total - done) if self.total and done else None
                emit(stage="infer", phase=self.phase, done=done, total=self.total, frac=frac, eta_s=eta,
                     msg=f"Reading the scroll: {done} of {self.total or '?'} patches")


def run(volume: str, out_prefix: Path, roi=None, direction="forward", threads=0, batch=0,
        overlap=0.5, loader_threads=4, checkpoint: Path | None = None) -> list[Path]:
    setup_paths()
    info = cpu_info()
    threads = threads or auto_threads(info)
    batch = batch or auto_batch(info)
    import torch
    torch.set_num_threads(threads)
    try:
        import vc_reader
    except ImportError as e:
        raise RuntimeError("vc_reader is missing from this install") from e
    from vesuvius.ink_detection.inference import infer

    ckpt = fetch_checkpoint(checkpoint)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    phases = ["forward", "reverse"] if direction == "both" else [direction]
    phase_iter = iter(phases)
    total_holder: dict = {}

    import logging
    class _Grab(logging.Handler):  # villa logs "Selected N patches for inference"
        def emit(self, record):
            try:
                if record.msg.startswith("Selected %d patches"):
                    total_holder["n"] = int(record.args[0])
            except Exception:
                pass
    logging.getLogger(infer.__name__).addHandler(_Grab())

    orig = infer.run_block_inference
    def wrapped(**kw):
        kw["loader"] = _Progress(kw["loader"], next(phase_iter, "forward"), total_holder.get("n"))
        return orig(**kw)
    infer.run_block_inference = wrapped

    argv = [volume, str(ckpt), str(out_prefix.with_suffix(".tif")), "--device", "cpu",
            "--overlap", str(overlap), "--blend-mode", "hann", "--direction", direction,
            "--batch-size", str(batch), "--no-compile"]
    if loader_threads > 0:
        # Threads instead of spawned worker processes: less RAM, no extra torch imports (slow on Windows).
        argv += ["--num-workers", "0", "--loader-threads", str(loader_threads)]
    if roi:
        argv += ["--roi", *map(str, roi)]
    emit(stage="start", msg=f"Using {threads} CPU threads, batch {batch}", threads=threads, batch=batch,
         torch=torch.__version__, backend="lite" if getattr(vc_reader, "LITE", False) else "native", cpu=info)
    t0 = time.time()
    infer.main(argv)
    secs = time.time() - t0
    tifs = [out_prefix.with_suffix(".tif")]
    if direction == "both":
        tifs.append(out_prefix.with_name(out_prefix.name + "_reverse.tif"))
    pngs = []
    for tif in tifs:
        if tif.exists():
            png = tif.with_suffix(".png")
            preview_png(tif, png)
            pngs.append(png)
    emit(stage="done", msg=f"Done in {secs/60:.1f} minutes", seconds=secs, images=[str(p) for p in pngs],
         tifs=[str(t) for t in tifs if t.exists()])
    return pngs


def exit_with_parent() -> None:
    """If the app window is closed, stop too (otherwise a closed window could leave the CPU busy)."""
    ppid = int(os.environ.get("VESUVIUS_READER_PARENT", "0") or 0)
    if not ppid:
        return
    try:
        import psutil
        parent = psutil.Process(ppid)
        created = parent.create_time()
    except Exception:
        return
    import threading
    def watch():
        while True:
            time.sleep(2)
            try:
                alive = parent.is_running() and parent.create_time() == created
            except Exception:
                alive = False
            if not alive:
                os._exit(3)
    threading.Thread(target=watch, daemon=True).start()


def main(argv=None) -> int:
    exit_with_parent()
    ap = argparse.ArgumentParser(description="Vesuvius Reader worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("volume")
    r.add_argument("--out", required=True)
    r.add_argument("--roi", nargs=4, type=int)
    r.add_argument("--direction", choices=("forward", "reverse", "both"), default="forward")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=0)
    r.add_argument("--overlap", type=float, default=0.5)
    r.add_argument("--loader-threads", type=int, default=4)
    r.add_argument("--checkpoint")
    p = sub.add_parser("preview")
    p.add_argument("volume")
    p.add_argument("--out", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--out", default=str(data_dir() / "selftest" / "selftest"))
    sub.add_parser("info")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            run(a.volume, Path(a.out), a.roi, a.direction, a.threads, a.batch_size, a.overlap,
                a.loader_threads, Path(a.checkpoint) if a.checkpoint else None)
        elif a.cmd == "preview":
            setup_paths()
            meta = segment_preview(a.volume, Path(a.out))
            emit(stage="preview", image=a.out, **meta)
        elif a.cmd == "selftest":
            pngs = run(SELFTEST_VOLUME, Path(a.out), SELFTEST_ROI)
            import numpy as np, tifffile
            t = tifffile.imread(str(Path(a.out).with_suffix(".tif")))
            ref = tifffile.imread(str(HERE / "selftest_reference.tif"))  # Linux CPU run, 2026-11-21
            ok = t.shape == ref.shape
            same = float((t == ref).mean()) if ok else 0.0
            maxdiff = int(np.abs(t.astype(int) - ref.astype(int)).max()) if ok else 255
            ok = ok and same >= 0.95 and maxdiff <= 2  # float rounding differs a little between CPUs/threads
            emit(stage="selftest", ok=bool(ok), shape=list(t.shape), identical_frac=same, max_diff=maxdiff,
                 images=[str(x) for x in pngs])
            return 0 if ok else 1
        elif a.cmd == "info":
            info = cpu_info()
            emit(stage="info", cpu=info, threads=auto_threads(info), batch=auto_batch(info), data_dir=str(data_dir()))
    except Exception as e:  # show a friendly message in the app, full traceback in the log
        import traceback
        traceback.print_exc()
        emit(stage="error", msg=f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
