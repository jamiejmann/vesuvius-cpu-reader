#!/usr/bin/env python3
"""Assemble the portable Windows build of Vesuvius Reader.

Result: dist/VesuviusReader/ and dist/VesuviusReader-win64-<version>.zip containing

    VesuviusReader/
      Vesuvius Reader.bat        <- double-click this
      README.txt
      python/                    <- official python.org "embeddable" Python (signed python.exe)
        Lib/site-packages/       <- torch (CPU), timm, zarr, ... preinstalled
      app/                       <- reader_core.py, vesuvius_reader_app.py, ui/
      vcr_lite/src/vc_reader/    <- pure-Python vc_reader stand-in
      villa/vesuvius/src/vesuvius/  <- ScrollPrize/villa inference code, pinned commit
      licenses/

Runs on Windows (GitHub Actions windows-latest) or cross-assembles from Linux/macOS with
`pip --platform win_amd64` (cannot run the self-test there). Needs git and pip on PATH.
"""
from __future__ import annotations

import argparse, io, os, platform, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # winapp/
PY_VERSION = "3.12.10"                               # last 3.12 release with Windows binaries
PY_TAG = "312"
VILLA_REPO = "https://github.com/ScrollPrize/villa.git"
VILLA_COMMIT = "c8f3ba4"                              # the commit read_segment_cpu.py was verified with
APP_VERSION = "1.0.0"
VC_DLLS = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll",
           "concrt140.dll", "vcomp140.dll"]

TRIM_DIRS = ["torch/include", "torch/share/cmake", "torch/testing/_internal/opinfo", "caffe2",
             "numpy/_core/tests", "numpy/tests", "numpy/f2py/tests", "PIL/tests", "zarr/testing"]
TRIM_GLOBS = ["torch/lib/*.lib", "**/*.pdb", "**/__pycache__"]


def sh(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def fetch(url: str) -> bytes:
    print("downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--villa-src", help="existing villa checkout to copy from (skips git clone)")
    ap.add_argument("--vcr-lite", default=str(next((p for p in (ROOT.parent / "vcr_lite", ROOT.parent / "release" / "vcr_lite")
                                                   if p.is_dir()), ROOT.parent / "vcr_lite")),
                    help="path to vcr_lite (default: ../vcr_lite in the repo)")
    ap.add_argument("--selftest", action="store_true", help="(Windows only) run a tiny end-to-end read after building")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    on_windows = platform.system() == "Windows"
    dist = Path(a.out); bundle = dist / "VesuviusReader"
    if bundle.exists():
        shutil.rmtree(bundle)
    py = bundle / "python"; py.mkdir(parents=True)

    # 1. embeddable Python
    z = fetch(f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip")
    zipfile.ZipFile(io.BytesIO(z)).extractall(py)
    (py / f"python{PY_TAG}._pth").write_text(
        f"python{PY_TAG}.zip\n.\nLib\\site-packages\n..\\app\n\n# site-packages + .pth files\nimport site\n")
    site = py / "Lib" / "site-packages"; site.mkdir(parents=True)

    # 2. dependencies, preinstalled (binary wheels only)
    reqs = ["--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple",
            "-r", ROOT / "requirements-winapp.txt"]
    native = on_windows and sys.version_info[:2] == (3, int(PY_TAG[1:]))
    if native:
        # CPU-only torch index first (same as the release README); everything else from PyPI.
        pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:", "--target", site, *reqs]
    else:
        # Cross-assembly: pip evaluates environment markers for the *build* machine (pulls Linux CUDA deps),
        # uv evaluates them for the target platform.
        uv = shutil.which("uv") or sys.exit("cross-building needs `uv` on PATH (pip cannot target Windows markers)")
        pip = [uv, "pip", "install", "--target", site, "--python-platform", "x86_64-pc-windows-msvc",
               "--python-version", f"3.{PY_TAG[1:]}", "--only-binary", ":all:", "--index-strategy", "unsafe-best-match",
               "--link-mode", "copy", *reqs]
    sh(*pip)
    # pip --target leaves console-script .exe launchers that point at the build machine's Python
    shutil.rmtree(site / "bin", ignore_errors=True)

    # 3. MSVC runtime next to python.exe so torch loads on PCs without the VC++ redistributable
    if on_windows:
        sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for dll in VC_DLLS:
            if (sysdir / dll).exists() and not (py / dll).exists():
                shutil.copy2(sysdir / dll, py / dll)

    # 4. our app + vcr_lite
    shutil.copytree(ROOT / "app", bundle / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(a.vcr_lite) / "src", bundle / "vcr_lite" / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))

    # 5. villa's `vesuvius` package at the pinned commit (only the Python sources)
    if a.villa_src:
        src = Path(a.villa_src)
    else:
        src = dist / "_villa"
        if src.exists():
            shutil.rmtree(src)
        sh("git", "clone", "--filter=blob:none", "--no-checkout", VILLA_REPO, src)
        sh("git", "-C", src, "sparse-checkout", "set", "vesuvius/src/vesuvius")
        sh("git", "-C", src, "checkout", VILLA_COMMIT)
    shutil.copytree(src / "vesuvius" / "src" / "vesuvius", bundle / "villa" / "vesuvius" / "src" / "vesuvius",
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "tests"))
    lic = bundle / "licenses"; lic.mkdir()
    shutil.copy2(src / "LICENSE", lic / "villa-LICENSE.txt") if (src / "LICENSE").exists() else None
    for p in (ROOT.parent / "LICENSE", ROOT.parent / "release" / "LICENSE"):
        if p.exists():
            shutil.copy2(p, lic / "vesuvius-cpu-reader-LICENSE.txt")
            break
    shutil.copy2(ROOT / "build" / "THIRD_PARTY.txt", lic / "THIRD_PARTY.txt")

    # 6. trim things nobody needs at run time
    for d in TRIM_DIRS:
        shutil.rmtree(site / d, ignore_errors=True)
    for g in TRIM_GLOBS:
        for p in list(site.glob(g)):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    # 7. launcher + readme
    (bundle / "Vesuvius Reader.bat").write_text(
        "@echo off\r\n"
        "title Vesuvius Reader\r\n"
        "cd /d \"%~dp0\"\r\n"
        "echo Starting Vesuvius Reader... your browser will open in a moment.\r\n"
        "\"%~dp0python\\python.exe\" -X utf8 \"%~dp0app\\vesuvius_reader_app.py\" %*\r\n"
        "if errorlevel 1 (\r\n  echo.\r\n  echo Something went wrong. Please copy the text above when you report it.\r\n  pause\r\n)\r\n",
        newline="")
    (bundle / "Self-test.bat").write_text(
        "@echo off\r\ncd /d \"%~dp0\"\r\n\"%~dp0python\\python.exe\" -X utf8 \"%~dp0app\\reader_core.py\" selftest\r\npause\r\n",
        newline="")
    shutil.copy2(ROOT / "build" / "README.txt", bundle / "README.txt")
    (bundle / "VERSION.txt").write_text(f"Vesuvius Reader {APP_VERSION}\nPython {PY_VERSION}\nvilla {VILLA_COMMIT}\n")

    files = sum(1 for p in bundle.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"bundle: {files} files, {size/1e6:.0f} MB unpacked", flush=True)

    if a.selftest:
        if not on_windows:
            sys.exit("--selftest needs Windows")
        sh(py / "python.exe", "-X", "utf8", bundle / "app" / "reader_core.py", "info")
        sh(py / "python.exe", "-X", "utf8", bundle / "app" / "reader_core.py", "selftest")

    if not a.no_zip:
        zpath = dist / f"VesuviusReader-win64-{APP_VERSION}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(bundle.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(dist))
        print(f"zip: {zpath} {zpath.stat().st_size/1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
