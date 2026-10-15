import unittest

import numpy as np
from scipy.sparse import csr_matrix

from prepare_data import asof_indices
from recommendation_models import evaluate, filter_known_unavailable


class AvailabilityTests(unittest.TestCase):
    def test_asof_join_uses_only_state_at_or_before_query_time(self):
        multiplier = 1_000
        property_items = np.array([10, 10, 20], dtype=np.int32)
        property_timestamps = np.array([100, 300, 200], dtype=np.int64)
        property_keys = property_items.astype(np.int64) * multiplier + property_timestamps
        property_keys.sort()

        result = asof_indices(
            property_keys,
            property_items,
            np.array([10, 10, 20, 30], dtype=np.int32),
            np.array([99, 200, 250, 250], dtype=np.int64),
            multiplier,
        )

        np.testing.assert_array_equal(result, [-1, 0, 2, -1])

    def test_filter_excludes_known_unavailable_and_keeps_unknown(self):
        availability = np.array([0, 1, -1, 1], dtype=np.int8)
        candidates = np.array([[0, 2]], dtype=np.int32)
        popular = np.array([0, 1, 2, 3], dtype=np.int32)

        result = filter_known_unavailable(candidates, availability, popular, 3)

        np.testing.assert_array_equal(result, [[2, 1, 3]])

    def test_metrics_report_unavailable_and_unknown_exposure(self):
        targets = csr_matrix(([1.0, 1.0], ([0, 0], [1, 3])), shape=(1, 4))
        recommendations = np.array([[0, 2]], dtype=np.int32)
        availability = np.array([0, 1, -1, 0], dtype=np.int8)

        result = evaluate(recommendations, np.array([0]), targets, availability, 2)

        self.assertEqual(result["recall@2"], 0.0)
        self.assertEqual(result["unavailable_recommendation_rate"], 0.5)
        self.assertEqual(result["unknown_availability_recommendation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
