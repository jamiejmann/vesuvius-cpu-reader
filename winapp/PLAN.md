# Vesuvius Reader for Windows: one-click app (December progress prize)

## Approach
**A portable folder built from the official python.org "embeddable" Python, with everything already installed, plus a small local web page as the window.**
The user unzips it and double-clicks `Vesuvius Reader.bat`. A console window starts a tiny server on 127.0.0.1 and the browser opens the app.

Why this approach and not PyInstaller or Nuitka:
- **Torch freezes badly.** PyTorch and timm read their own `.py` sources, load DLLs lazily, and import inductor/sympy at run time. PyInstaller and Nuitka need hooks and patches for all of that. In the embeddable bundle everything stays a plain Python install, so it works the same way it already works on Jamie's laptop.
- **Fewer antivirus problems.** Unsigned PyInstaller bootloaders are a well-known source of Defender false positives. Here `python.exe` is PSF-signed. The `.bat` file may still get one "Windows protected your PC" prompt, and the README explains how to get past it.
- **Fixable without rebuilding.** Every file is readable `.py`, so a tester can patch one line.
- **The browser page needs no extra packages.** Embeddable Python ships without tkinter. A browser page shows big images with zoom, needs no GUI library, and looks friendlier.

## Layout
```
app/reader_core.py           worker: model download (sha256), threads/batch sizing, villa inference, progress JSON, preview, selftest
app/vesuvius_reader_app.py   stdlib HTTP server + job control (start/stop/kill-tree, exits with parent), segment catalog
app/ui/index.html            the page: pick segment (736 from metadata.min.json; ~9 um filter -> 212), drag-box ROI on preview,
                             progress bar + ETA, result, side-by-side crop of the team's published map
app/selftest_reference.tif   384x768 reference for the self-test (Linux CPU run)
build/build_bundle.py        assembles dist/VesuviusReader + zip (native on Windows, or cross from Linux with uv)
.github/workflows/windows-app.yml   windows-latest: unit tests, build, selftest, server E2E, unzip-to-path-with-spaces, artifact, draft release on tag
tests/test_units.py          no-network unit tests
tests/smoke_server.py        browser-equivalent end-to-end test of the server
requirements-winapp.txt      pinned versions (same as the verified Linux env)
```
In the repo: sources in `winapp/`, workflow at `.github/workflows/windows-app.yml` (runs on pushes to the `winapp` branch and on demand).

## Sizes (measured from a cross-built bundle, win_amd64 wheels)
- Zip download: **about 191 MB**. Unzipped: **about 622 MB** in about 9,000 files (torch is 384 MB of that).
- The first run also downloads the Reader V3 checkpoint (138 MB, sha256-checked) into `%LOCALAPPDATA%\VesuviusReader`.
- Images are saved to `Documents\Vesuvius Reader\<segment>_<time>\`, with `last-run-log.txt` alongside.
