import json
import os
import tempfile
import unittest


def _has_tensorneko_lib():
    try:
        import tensorneko_lib.evaluation

        return hasattr(tensorneko_lib.evaluation, "ap_1d")
    except (ImportError, AttributeError):
        return False


@unittest.skipUnless(_has_tensorneko_lib(), "tensorneko_lib (Rust) not installed")
class TestLoc1dAP(unittest.TestCase):
    def setUp(self):
        from tensorneko_lib.evaluation import ap_1d, ar_1d, ap_ar_1d

        self.ap_1d = ap_1d
        self.ar_1d = ar_1d
        self.ap_ar_1d = ap_ar_1d
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def _write(self, name, obj):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    def _make_main_fixtures(self):
        labels = [
            {"file": "video_001.mp4", "segments": [[1.0, 3.0], [5.0, 7.0]]},
            {"file": "video_002.mp4", "segments": [[2.0, 4.0]]},
            {
                "file": "video_003.mp4",
                "segments": [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]],
            },
        ]
        proposals = {
            "video_001.mp4": [[0.9, 1.0, 3.0], [0.8, 5.5, 7.5], [0.3, 10.0, 12.0]],
            "video_002.mp4": [[0.95, 2.0, 4.5], [0.4, 8.0, 9.0]],
            "video_003.mp4": [
                [0.85, 0.5, 2.5],
                [0.7, 3.5, 6.5],
                [0.6, 7.5, 10.5],
                [0.2, 15.0, 17.0],
            ],
        }
        lp = self._write("labels.json", labels)
        pp = self._write("proposals.json", proposals)
        return pp, lp

    def test_ap_1d_main_scenario(self):
        pp, lp = self._make_main_fixtures()
        result = self.ap_1d(pp, lp, "file", "segments", 1.0, [0.3, 0.5, 0.7])

        self.assertEqual(len(result), 3)
        for key in result:
            self.assertIsInstance(result[key], float)

        vals = {round(k, 1): v for k, v in result.items()}
        self.assertAlmostEqual(vals[0.3], 0.8333333, places=4)
        self.assertAlmostEqual(vals[0.5], 0.8333333, places=4)
        self.assertAlmostEqual(vals[0.7], 0.1666667, places=4)

    def test_ar_1d_main_scenario(self):
        pp, lp = self._make_main_fixtures()
        result = self.ar_1d(pp, lp, "file", "segments", 1.0, [1, 3, 5], [0.3, 0.5, 0.7])

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1], 0.4444444, places=4)
        self.assertAlmostEqual(result[3], 0.7777777, places=4)
        self.assertAlmostEqual(result[5], 0.7777777, places=4)

    def test_ap_ar_1d_consistency(self):
        pp, lp = self._make_main_fixtures()
        ap = self.ap_1d(pp, lp, "file", "segments", 1.0, [0.3, 0.5])
        ar = self.ar_1d(pp, lp, "file", "segments", 1.0, [1, 3], [0.3, 0.5])
        combined = self.ap_ar_1d(
            pp, lp, "file", "segments", 1.0, [0.3, 0.5], [1, 3], [0.3, 0.5]
        )

        self.assertEqual(combined["ap"], ap)
        self.assertEqual(combined["ar"], ar)

    def test_ar_perfect_recall(self):
        labels = [{"file": "v1.mp4", "segments": [[1.0, 3.0], [5.0, 7.0]]}]
        proposals = {"v1.mp4": [[0.9, 1.0, 3.0], [0.8, 5.0, 7.0]]}
        lp = self._write("l_perf.json", labels)
        pp = self._write("p_perf.json", proposals)

        result = self.ar_1d(pp, lp, "file", "segments", 1.0, [2], [0.5])
        self.assertAlmostEqual(result[2], 1.0, places=4)

    def test_ar_single_proposal_partial(self):
        labels = [{"file": "v1.mp4", "segments": [[1.0, 3.0], [5.0, 7.0]]}]
        proposals = {"v1.mp4": [[0.9, 1.0, 3.0], [0.8, 5.0, 7.0]]}
        lp = self._write("l_partial.json", labels)
        pp = self._write("p_partial.json", proposals)

        result = self.ar_1d(pp, lp, "file", "segments", 1.0, [1], [0.5])
        self.assertAlmostEqual(result[1], 0.5, places=4)

    def test_fps_scaling(self):
        labels = [{"file": "v1.mp4", "segments": [[1.0, 3.0]]}]
        proposals_fps30 = {"v1.mp4": [[0.9, 30.0, 90.0]]}
        lp = self._write("l_fps.json", labels)
        pp = self._write("p_fps.json", proposals_fps30)

        result = self.ar_1d(pp, lp, "file", "segments", 30.0, [1], [0.5])
        self.assertAlmostEqual(result[1], 1.0, places=4)

    def test_no_matching_proposals(self):
        labels = [{"file": "v1.mp4", "segments": [[0.0, 1.0]]}]
        proposals = {"v1.mp4": [[1.0, 100.0, 200.0]]}
        lp = self._write("l_no.json", labels)
        pp = self._write("p_no.json", proposals)

        result = self.ar_1d(pp, lp, "file", "segments", 1.0, [1], [0.5])
        self.assertAlmostEqual(result[1], 0.0, places=4)

    def test_multiple_iou_thresholds_ordering(self):
        pp, lp = self._make_main_fixtures()
        result = self.ap_1d(pp, lp, "file", "segments", 1.0, [0.3, 0.5, 0.7])

        vals = sorted(result.items(), key=lambda x: x[0])
        self.assertGreaterEqual(vals[0][1], vals[-1][1])

    def test_increasing_n_proposals_monotonic_ar(self):
        pp, lp = self._make_main_fixtures()
        result = self.ar_1d(pp, lp, "file", "segments", 1.0, [1, 2, 3, 5], [0.5])

        ar_values = [result[n] for n in [1, 2, 3, 5]]
        for i in range(len(ar_values) - 1):
            self.assertLessEqual(ar_values[i], ar_values[i + 1] + 1e-6)


if __name__ == "__main__":
    unittest.main()
