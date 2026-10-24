import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from worldlines_motion.workbench.server import Handler


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return json.load(r)

    def test_health_clips_and_index(self):
        h = self.get("/api/health")
        self.assertIn("mediapipe", h)
        clips = self.get("/api/clips")
        self.assertTrue(any(c["name"] == "walk_squat.smplx.npz" for c in clips))
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10) as r:
            self.assertIn(b"worldlines motion", r.read())

    def test_naturalize_endpoint(self):
        h = self.get("/api/health")
        if not h.get("smplx"):
            self.skipTest("no SMPL-X model")
        c = self.get("/api/clip?path=demos/clips/walk_squat.smplx.npz")
        body = json.dumps({"clip": c, "config": {"smooth_sigma": 1.0}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/naturalize", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        self.assertEqual(out["n"], c["n"])
        self.assertLess(out["metrics"]["floor_penetration_m"], 0.02)


if __name__ == "__main__":
    unittest.main()
