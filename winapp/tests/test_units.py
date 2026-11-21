"""Fast unit tests that need no network and no torch:  python -m unittest tests/test_units.py"""
import os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
os.environ.setdefault("VESUVIUS_READER_HOME", tempfile.mkdtemp())
os.environ.setdefault("VESUVIUS_READER_OUT", tempfile.mkdtemp())
import reader_core as core  # noqa: E402
import vesuvius_reader_app as app  # noqa: E402

META = {"samples": {"PHerc1447": {
    "sample": {"properties": {"type": "scroll"}},
    "volumes": {"V9": {"properties": {"pixel_size_um": 8.64}}, "V2": {"properties": {"pixel_size_um": 2.4}}},
    "segments": {"1": {"suffix": "w058", "long_id": "1-w058", "data": [
        {"type": "layers-zarr", "origins": [{"path": "PHerc1447/segments/1-w058/sv/a.zarr/"}], "parameters": {"target_volume": "V9"}},
        {"type": "layers-zarr", "origins": [{"path": "PHerc1447/segments/1-w058/sv/b.zarr/"}], "parameters": {"target_volume": "V2"}},
        {"type": "ink-detection-downsampled", "origins": [{"path": "PHerc1447/x-ds8.jpg"}], "parameters": {"target_volume": "V9"}},
    ]}}}}}


class T(unittest.TestCase):
    def test_catalog(self):
        cat = app.build_catalog(META)
        self.assertEqual(len(cat), 2)
        a = [c for c in cat if c["pixel_um"] == 8.64][0]
        self.assertTrue(a["good_res"])
        self.assertEqual(a["url"], "s3://vesuvius-challenge-open-data/PHerc1447/segments/1-w058/sv/a.zarr")
        self.assertTrue(a["official_preview"].endswith("PHerc1447/x-ds8.jpg"))
        b = [c for c in cat if c["pixel_um"] == 2.4][0]
        self.assertFalse(b["good_res"]); self.assertIsNone(b["official_preview"])

    def test_threads(self):
        self.assertEqual(core.auto_threads({"logical": 8, "physical": 4, "idle": None, "avail_gb": None}), 4)
        self.assertEqual(core.auto_threads({"logical": 8, "physical": 4, "idle": 2.1, "avail_gb": 8}), 2)
        self.assertEqual(core.auto_threads({"logical": 8, "physical": 4, "idle": 0.1, "avail_gb": 8}), 1)
        self.assertEqual(core.auto_batch({"avail_gb": 2.0}), 1)
        self.assertEqual(core.auto_batch({"avail_gb": 8.0}), 2)

    def test_validate(self):
        self.assertTrue(app.validate_volume(" s3://b/x.zarr ").startswith("s3://"))
        with self.assertRaises(ValueError):
            app.validate_volume("")
        with self.assertRaises(ValueError):
            app.validate_volume("/definitely/not/here.zarr")
        d = tempfile.mkdtemp()
        self.assertEqual(app.validate_volume(f'"{d}"'), str(Path(d).resolve()))

    def test_safe_name(self):
        self.assertEqual(app.safe_name("PHerc1447 w058/../x"), "PHerc1447_w058_.._x")
        self.assertEqual(app.safe_name("///"), "segment")


if __name__ == "__main__":
    unittest.main()
