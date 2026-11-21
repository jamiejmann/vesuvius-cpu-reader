#!/usr/bin/env python3
"""End-to-end smoke test of the app server, as the browser would use it (stdlib only).

    python tests/smoke_server.py [--python path\\to\\python.exe] [--app path\\to\\app]

Starts vesuvius_reader_app.py on a free port, checks the security guard, loads the segment list,
makes a preview, reads a tiny area of PHerc1447 w058, checks the PNG is served, tests Stop on a
whole-segment run, then quits the server. Exits non-zero on any failure.
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

W058 = ("s3://vesuvius-challenge-open-data/PHerc1447/segments/20260928183143-w058/"
        "surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--app", default=str(Path(__file__).resolve().parents[1] / "app"))
    a = ap.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    base = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([a.python, str(Path(a.app) / "vesuvius_reader_app.py"), "--port", str(port), "--no-browser"])

    def call(path, body=None, headers=None, expect=200):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                code, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, data = e.code, e.read()
        assert code == expect, f"{path}: HTTP {code} (wanted {expect}): {data[:300]!r}"
        try:
            return json.loads(data)
        except ValueError:
            return data

    def wait_done(limit):
        t0 = time.time()
        while time.time() - t0 < limit:
            st = call("/api/status")
            if st["state"] != "running":
                return st
            time.sleep(2)
        raise AssertionError("run did not finish in time")

    try:
        for _ in range(60):
            try:
                info = call("/api/info"); break
            except Exception:
                time.sleep(0.5)
        print("info:", info)
        call("/api/info", headers={"Host": "evil.example"}, expect=403)
        call("/api/stop", {}, headers={"Origin": "http://evil.example"}, expect=403)
        segs = call("/api/segments")["segments"]
        assert any(x["url"] == W058 for x in segs), "w058 missing from catalog"
        print("segments:", len(segs))
        pv = call("/api/preview", {"volume": W058})
        assert pv.get("stage") == "preview" and pv["full_w"] == 8080, pv
        call("/file?path=" + urllib.request.quote(pv["image"]))
        call("/file?path=" + urllib.request.quote(str(Path(a.app) / "reader_core.py")), expect=404)
        call("/api/run", {"volume": "Z:/does/not/exist.zarr"}, expect=400)
        r = call("/api/run", {"volume": W058, "roi": [2200, 2584, 2200, 2968], "label": "smoke"})
        print("run:", r)
        st = wait_done(900)
        assert st["state"] == "done" and st["images"], st
        png = call("/file?path=" + urllib.request.quote(st["images"][0]))
        assert isinstance(png, bytes) and png[:4] == b"\x89PNG"
        print("tiny read OK in", round(st["elapsed_s"]), "s:", st["images"][0])
        call("/api/run", {"volume": W058, "label": "smoke_stop"})
        time.sleep(8)
        call("/api/run", {"volume": W058}, expect=400)  # already running
        call("/api/stop", {})
        time.sleep(3)
        assert call("/api/status")["state"] == "stopped"
        print("stop OK")
        call("/api/quit", {})
        srv.wait(timeout=20)
        print("SMOKE TEST PASSED")
        return 0
    finally:
        if srv.poll() is None:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())
