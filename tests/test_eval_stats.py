"""The report's statistics, offline: pinned Wilson and pass@k figures, percentiles and the seeded bootstrap."""
import contextlib
import os
import random
import tempfile
import unittest
from unittest import mock

from evals import stats
from tests.fake_claude import tripwire

_trip = {}


def setUpModule():
    # The username rewrite is machine-dependent, so pin a name no fixture word can collide with.
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    _trip["log"] = tripwire(tmp.name)
    patcher = mock.patch.dict(os.environ, {"USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser",
                                           "PATH": tmp.name + os.pathsep + os.environ.get("PATH", "")})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def tearDownModule():
    if _trip["log"].exists():
        raise AssertionError(f"a test reached `claude`: {_trip['log'].read_text()}")


MIXED = [([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), ([1, 1, 0], [0, 0, 1, 0])]


class WilsonTest(unittest.TestCase):
    def test_pinned_intervals(self):
        for c, n, expected in [(8, 10, (0.4902, 0.9433)), (0, 10, (0.0, 0.2775)), (10, 10, (0.7225, 1.0)),
                               (0, 3, (0.0, 0.5615)), (3, 3, (0.4385, 1.0))]:
            with self.subTest(c=c, n=n):
                self.assertEqual(stats.wilson(c, n), expected)

    def test_the_bounds_are_clamped(self):
        # Unclamped, 0/7 gives -0.0 (which equals 0.0 but prints "-0.0000") and some n give a bound above 1.
        for n in (7, 9, 14, 17, 18):
            with self.subTest(n=n):
                self.assertEqual(repr(stats.wilson(0, n)[0]), "0.0")
                self.assertEqual(repr(stats.wilson(n, n)[1]), "1.0")

    def test_no_trials_has_no_interval(self):
        self.assertIsNone(stats.wilson(0, 0))

    def test_z_is_the_pinned_value(self):
        self.assertEqual(stats.Z, 1.959964)


class PassAtKTest(unittest.TestCase):
    def test_pinned_values_for_8_of_10(self):
        self.assertEqual([stats.pass_at_k(10, 8, k) for k in stats.KS], [0.8, 1.0, 1.0])
        self.assertEqual([stats.pass_pow_k(10, 8, k) for k in stats.KS], [0.8, 0.4667, 0.2222])

    def test_2_of_3_at_k_3(self):
        self.assertEqual((stats.pass_at_k(3, 2, 3), stats.pass_pow_k(3, 2, 3)), (1.0, 0.0))

    def test_k_beyond_the_denominator_is_na(self):
        self.assertEqual([stats.pass_at_k(3, 2, k) for k in stats.KS], [0.6667, 1.0, None])
        self.assertEqual([stats.pass_pow_k(3, 2, k) for k in stats.KS], [0.6667, 0.0, None])
        for k in stats.KS:
            self.assertIsNone(stats.pass_at_k(0, 0, k))
            self.assertIsNone(stats.pass_pow_k(0, 0, k))

    def test_no_passes_and_all_passes(self):
        self.assertEqual([stats.pass_at_k(10, 0, k) for k in stats.KS], [0.0, 0.0, 0.0])
        self.assertEqual([stats.pass_pow_k(10, 0, k) for k in stats.KS], [0.0, 0.0, 0.0])
        self.assertEqual([stats.pass_at_k(10, 10, k) for k in stats.KS], [1.0, 1.0, 1.0])
        self.assertEqual([stats.pass_pow_k(10, 10, k) for k in stats.KS], [1.0, 1.0, 1.0])


class PercentileTest(unittest.TestCase):
    def test_one_value_is_every_percentile(self):
        self.assertEqual((stats.percentile([7], 50), stats.percentile([7], 95)), (7, 7))

    def test_ten_values_by_nearest_rank(self):
        values = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]
        self.assertEqual((stats.percentile(values, 50), stats.percentile(values, 95)), (5, 10))

    def test_p95_is_the_max_below_20_values_and_not_at_20(self):
        self.assertEqual(stats.percentile(list(range(1, 20)), 95), 19)
        self.assertEqual(stats.percentile(list(range(1, 21)), 95), 19)

    def test_empty_is_none(self):
        self.assertIsNone(stats.percentile([], 50))


class BootstrapTest(unittest.TestCase):
    def test_pinned_results_on_literal_input(self):
        self.assertEqual(stats.bootstrap_ci(MIXED[:1], seed=7, resamples=1000), (0.1, 0.8))
        self.assertEqual(stats.bootstrap_ci(MIXED, seed=7, resamples=1000), (0.0667, 0.8))
        self.assertEqual(stats.bootstrap_ci(MIXED, seed=7, resamples=10_000), (0.05, 0.8))

    def test_the_interval_endpoints_are_the_plan_s_order_statistics(self):
        # Chosen so that, at B=999, the neighbouring order statistics on both sides differ: d[floor(.025B)] is the
        # low end and d[ceil(.975B) - 1] the high end; the neighbours are 0.1446/0.1476 and 0.6225/0.626.
        pairs = [([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
                 ([1, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]),
                 ([1, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])]
        self.assertEqual(stats.bootstrap_ci(pairs, seed=1, resamples=999), (0.1463, 0.6241))

    def test_the_high_endpoint_is_pinned_where_975B_is_whole(self):
        pairs = [([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
                 ([1, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]),
                 ([1, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])]
        self.assertEqual(stats.bootstrap_ci(pairs, seed=1, resamples=1000), (0.1476, 0.6225))

    def test_same_seed_same_interval_other_seed_another(self):
        a = stats.bootstrap_ci(MIXED, seed=1, resamples=1000)
        self.assertEqual(a, stats.bootstrap_ci(MIXED, seed=1, resamples=1000))
        self.assertNotEqual(a, stats.bootstrap_ci(MIXED, seed=2, resamples=1000))

    def test_identical_cells_have_an_interval_containing_zero(self):
        vector = [1, 0, 1, 1, 0, 1]
        lo, hi = stats.bootstrap_ci([(vector, list(vector))], seed=3, resamples=1000)
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)

    def test_all_pass_against_all_fail_is_exactly_one(self):
        self.assertEqual(stats.bootstrap_ci([([1] * 5, [0] * 5)], seed=3, resamples=1000), (1.0, 1.0))
        self.assertEqual(stats.bootstrap_ci([([0] * 5, [1] * 5)], seed=3, resamples=1000), (-1.0, -1.0))

    def test_an_empty_side_is_na(self):
        self.assertIsNone(stats.bootstrap_ci([([1, 0], [])], seed=3, resamples=100))
        self.assertIsNone(stats.bootstrap_ci([([], [1, 0])], seed=3, resamples=100))
        self.assertIsNone(stats.delta([1, 0], []))
        self.assertIsNone(stats.delta([], [1, 0]))

    def test_a_case_with_an_empty_side_is_left_out_of_the_interval(self):
        with_empty = stats.bootstrap_ci([MIXED[0], ([1, 1], [])], seed=7, resamples=1000)
        self.assertEqual(with_empty, stats.bootstrap_ci(MIXED[:1], seed=7, resamples=1000))

    def test_the_bootstrap_uses_only_random(self):
        names = ("choices", "sample", "randrange", "randint", "shuffle")
        with contextlib.ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(random.Random, name, side_effect=AssertionError(name)))
                stack.enter_context(mock.patch.object(random, name, side_effect=AssertionError(name)))
            self.assertEqual(stats.bootstrap_ci(MIXED[:1], seed=7, resamples=1000), (0.1, 0.8))


class DeltaTest(unittest.TestCase):
    def test_point_delta_is_the_rate_difference(self):
        self.assertEqual(stats.delta([1, 1, 1, 0], [1, 0, 0, 0]), 0.5)

    def test_overall_is_the_mean_of_the_per_case_deltas(self):
        self.assertEqual(stats.overall_delta(MIXED), (0.4583, 2))

    def test_overall_with_one_case_dropped(self):
        self.assertEqual(stats.overall_delta([MIXED[0], ([1, 1], [])]), (0.5, 1))

    def test_overall_with_every_case_dropped(self):
        self.assertEqual(stats.overall_delta([([], [1]), ([1], [])]), (None, 0))
        self.assertEqual(stats.overall_delta([]), (None, 0))
        self.assertIsNone(stats.bootstrap_ci([([], [1]), ([1], [])], seed=1, resamples=100))


if __name__ == "__main__":
    unittest.main()
