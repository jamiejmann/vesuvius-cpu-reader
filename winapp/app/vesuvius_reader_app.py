#!/usr/bin/env python3
"""Vesuvius Reader - a small local app around Reader V3 for people without a GPU.

Double-click "Vesuvius Reader.bat" (Windows) or run `python vesuvius_reader_app.py`.
It starts a tiny web server on this computer only (127.0.0.1) and opens it in your browser.
Nothing is sent anywhere except the scroll-data downloads from the public Vesuvius Challenge
bucket and the one-time model download from Hugging Face. Uses only the Python standard
library here; the heavy lifting happens in reader_core.py in a separate process.
"""
from __future__ import annotations

import gzip, json, os, re, socket, subprocess, sys, threading, time, urllib.request, webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reader_core as core  # noqa: E402  (light import: no torch at module level)

APP_VERSION = "0.1.0"
BUCKET = "vesuvius-challenge-open-data"
BUCKET_HTTPS = f"https://{BUCKET}.s3.amazonaws.com/"
META_URL = BUCKET_HTTPS + "metadata.min.json"
GOOD_UM = (7.5, 10.0)  # Reader V3 is a ~9 um model; these segments are the right resolution


def output_dir() -> Path:
    env = os.environ.get("VESUVIUS_READER_OUT")
    if env:
        p = Path(env)
    else:
        docs = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
        p = (docs if docs.is_dir() else Path.home()) / "Vesuvius Reader"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- segment catalog
def _https(path: str) -> str:
    return BUCKET_HTTPS + path.lstrip("/")


def build_catalog(meta: dict) -> list[dict]:
    out = []
    for scroll, sm in sorted(meta.get("samples", {}).items()):
        vols = sm.get("volumes") or {}
        kind = ((sm.get("sample") or {}).get("properties") or {}).get("type")
        for sid, seg in (sm.get("segments") or {}).items():
            data = seg.get("data") or []
            previews = {}
            for e in data:
                if e.get("type") == "ink-detection-downsampled":
                    tv = (e.get("parameters") or {}).get("target_volume")
                    previews.setdefault(tv, _https(e["origins"][0]["path"]))
            for e in data:
                if e.get("type") != "layers-zarr":
                    continue
                tv = (e.get("parameters") or {}).get("target_volume")
                um = ((vols.get(tv) or {}).get("properties") or {}).get("pixel_size_um")
                path = e["origins"][0]["path"].rstrip("/")
                out.append({
                    "scroll": scroll, "kind": kind, "segment": seg.get("long_id") or sid,
                    "name": seg.get("suffix") or sid, "pixel_um": um,
                    "good_res": bool(um and GOOD_UM[0] <= um <= GOOD_UM[1]),
                    "url": f"s3://{BUCKET}/{path}", "official_preview": previews.get(tv),
                })
    return out


_catalog_lock = threading.Lock()


def get_catalog(force: bool = False) -> list[dict]:
    cache = core.data_dir() / "catalog.json"
    with _catalog_lock:
        if not force and cache.exists() and time.time() - cache.stat().st_mtime < 86400:
            return json.loads(cache.read_text())
        req = urllib.request.Request(META_URL, headers={"User-Agent": "vesuvius-reader", "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        cat = build_catalog(json.loads(raw))
        cache.write_text(json.dumps(cat))
        return cat


# ---------------------------------------------------------------- worker process management
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.state = "idle"  # idle | running | done | error | stopped
        self.last: dict = {}
        self.log: list[str] = []
        self.images: list[str] = []
        self.started = 0.0
        self.name = ""

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "last": self.last, "log": self.log[-40:], "images": self.images,
                    "elapsed_s": (time.time() - self.started) if self.started else 0, "name": self.name}

    def start(self, args: list[str], name: str) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise ValueError("Already reading a segment. Press Stop first.")
            self.state, self.last, self.log, self.images = "running", {"msg": "Starting..."}, [], []
            self.started, self.name = time.time(), name
            self.proc = spawn_worker(args)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        logf = open(output_dir() / "last-run-log.txt", "w", encoding="utf-8")
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\r\n")
            logf.write(line + "\n"); logf.flush()
            with self.lock:
                if line.startswith("@@ "):
                    try:
                        msg = json.loads(line[3:])
                    except ValueError:
                        continue
                    self.last = msg
                    if msg.get("stage") == "done":
                        self.images = msg.get("images", [])
                    if msg.get("stage") == "error":
                        self.state = "error"
                else:
                    self.log.append(line)
                    self.log = self.log[-400:]
        code = proc.wait()
        logf.close()
        with self.lock:
            if self.state == "running":
                self.state = "done" if code == 0 and self.images else "error"
                if self.state == "error" and self.last.get("stage") != "error":
                    self.last = {"stage": "error", "msg": "The reader stopped unexpectedly. See last-run-log.txt in the output folder."}

    def stop(self) -> None:
        with self.lock:
            p = self.proc
            if p and p.poll() is None:
                self.state = "stopped"
                self.last = {"stage": "stopped", "msg": "Stopped."}
        if p and p.poll() is None:
            kill_tree(p)


def spawn_worker(args: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, "-u", str(HERE / "reader_core.py"), *args]
    kw: dict = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                         "VESUVIUS_READER_PARENT": str(os.getpid())})
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def kill_tree(p: subprocess.Popen) -> None:
    """Stop the worker and anything it started (loader threads die with it)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            import signal
            os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        p.kill()


JOB = Job()


def run_preview(volume: str) -> dict:
    out = core.data_dir() / "previews" / (re.sub(r"[^A-Za-z0-9._-]+", "_", volume)[-120:] + ".png")
    if out.exists() and (out.with_suffix(".json")).exists():
        return json.loads(out.with_suffix(".json").read_text())
    p = spawn_worker(["preview", volume, "--out", str(out)])
    text, _ = p.communicate(timeout=300)
    for line in text.splitlines()[::-1]:
        if line.startswith("@@ "):
            msg = json.loads(line[3:])
            if msg.get("stage") == "preview":
                out.with_suffix(".json").write_text(json.dumps(msg))
            return msg
    return {"stage": "error", "msg": "Could not make a preview. " + text[-500:]}


def windows_folder_dialog() -> str | None:
    if os.name != "nt":
        return None
    ps = ("Add-Type -AssemblyName System.Windows.Forms;"
          "$f=New-Object System.Windows.Forms.FolderBrowserDialog;"
          "$f.Description='Pick a surface-volume .zarr folder';"
          "if($f.ShowDialog() -eq 'OK'){Write-Output $f.SelectedPath}")
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return r.stdout.strip() or None


def open_in_explorer(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80] or "segment"


def validate_volume(v: str) -> str:
    v = (v or "").strip().strip('"')
    if not v:
        raise ValueError("Pick a segment or enter a path first.")
    if v.startswith(("s3://", "https://", "http://")):
        return v
    p = Path(v)
    if not p.exists():
        raise ValueError(f"That folder does not exist: {v}")
    return str(p.resolve())


def start_run(body: dict) -> dict:
    volume = validate_volume(body.get("volume", ""))
    roi = body.get("roi")
    if roi:
        roi = [int(x) for x in roi]
        if len(roi) != 4 or roi[1] <= roi[0] or roi[3] <= roi[2]:
            raise ValueError("The selected area is empty; drag a rectangle on the preview.")
    label = safe_name(body.get("label") or Path(volume.rstrip("/")).name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = output_dir() / f"{label}_{stamp}" / label
    args = ["run", volume, "--out", str(out), "--direction", body.get("direction", "forward")]
    if roi:
        args += ["--roi", *map(str, roi)]
    for k, flag in (("threads", "--threads"), ("batch", "--batch-size")):
        if body.get(k):
            args += [flag, str(int(body[k]))]
    JOB.start(args, label)
    return {"ok": True, "out": str(out.parent)}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "VesuviusReader/" + APP_VERSION

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _local_only(self) -> bool:
        # Refuse requests from other websites (DNS rebinding / CSRF): Host must be our loopback address.
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self._send(403, b"forbidden", "text/plain")
            return False
        origin = self.headers.get("Origin")
        if origin and not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
            self._send(403, b"forbidden", "text/plain")
            return False
        return True

    def do_GET(self):
        if not self._local_only():
            return
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (HERE / "ui" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/info":
                info = core.cpu_info()
                self._json({"version": APP_VERSION, "cpu": info, "threads": core.auto_threads(info),
                            "batch": core.auto_batch(info), "out_dir": str(output_dir()),
                            "model_ready": (core.data_dir() / "reader-v3.pth").exists(), "windows": os.name == "nt"})
            elif u.path == "/api/segments":
                force = parse_qs(u.query).get("refresh") == ["1"]
                self._json({"segments": get_catalog(force)})
            elif u.path == "/api/status":
                self._json(JOB.snapshot())
            elif u.path == "/file":
                p = Path(parse_qs(u.query).get("path", [""])[0]).resolve()
                roots = [output_dir().resolve(), core.data_dir().resolve()]
                if p.suffix.lower() not in (".png", ".jpg") or not any(r in p.parents for r in roots) or not p.is_file():
                    return self._send(404, b"not found", "text/plain")
                self._send(200, p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if not self._local_only():
            return
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/preview":
                self._json(run_preview(validate_volume(body.get("volume", ""))))
            elif u.path == "/api/run":
                self._json(start_run(body))
            elif u.path == "/api/stop":
                JOB.stop(); self._json({"ok": True})
            elif u.path == "/api/browse":
                self._json({"path": windows_folder_dialog()})
            elif u.path == "/api/open":
                target = Path(body.get("path") or output_dir())
                target = target if target.is_dir() else target.parent
                if output_dir().resolve() in [target.resolve(), *target.resolve().parents]:
                    open_in_explorer(target)
                self._json({"ok": True})
            elif u.path == "/api/quit":
                self._json({"ok": True})
                JOB.stop()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send(404, b"not found", "text/plain")
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError("no free port")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    port = a.port or free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Vesuvius Reader {APP_VERSION} is running at {url}")
    print("Keep this window open while you use the app. Close it (or press Ctrl+C) to quit.")
    print(f"Your images are saved in: {output_dir()}")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
