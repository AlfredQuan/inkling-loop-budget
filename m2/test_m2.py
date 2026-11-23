import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

from fit import confidence_features, policy, summarize, weighted_threshold, winning_choice


spec = importlib.util.spec_from_file_location("m2_route", Path(__file__).with_name("route.py"))
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


class EvaluationTest(unittest.TestCase):
    def test_ties_are_never_partially_admitted(self):
        scores = np.array([-.5, .8, .3, .8])
        tokens = np.array([10, 20, 40, 30])
        self.assertFalse(policy(scores, weighted_threshold(scores, tokens, .49)).any())
        self.assertEqual(policy(scores, weighted_threshold(scores, tokens, .5)).tolist(), [False, True, False, True])
        self.assertTrue(policy(scores, weighted_threshold(scores, tokens, 1)).all())

    def test_signed_reward_and_replay_are_counted_per_case(self):
        outcomes = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
        m = summarize(outcomes, np.array([True, True, False, False]), np.array([10, 20, 30, 40]), np.array(["a"] * 4))
        self.assertEqual((m["accuracy"], m["rescued"], m["missed_rescue"], m["harmed_by_extra_passes"]),
                         (.5, 1, 1, 1))
        self.assertEqual(m["rescue_recall"], .5)
        self.assertAlmostEqual(m["replay_work_saving"], .2)
        self.assertAlmostEqual(m["reuse_proxy_saving"], .3125)

    def test_runtime_uses_only_first_pass_inputs(self):
        case = {"logits": [5., 3., -4., 0.], "tokens": 81, "state": [.25] * 96}
        conf = confidence_features(np.array([case["logits"]]), np.array([case["tokens"]]))[0]
        coef = [.1, .2, -.3, .4] + [0.] * 95 + [.8]
        expected = .62 + np.dot(coef[:4], conf) + .2
        head = {"coefficients": coef, "intercept": .62, "threshold": expected - .01}
        out = inference.route(head, case)
        self.assertEqual(out["passes"], 3)
        self.assertTrue(math.isclose(out["score"], expected, abs_tol=1e-12))
        with self.assertRaises(ValueError):
            inference.route(head, dict(case, state=case["state"][:-1]))
        with self.assertRaises(ValueError):
            winning_choice([0, 0, 0, float("nan")])


if __name__ == "__main__":
    unittest.main()
