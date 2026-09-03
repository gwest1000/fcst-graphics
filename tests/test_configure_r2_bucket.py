from __future__ import annotations

import unittest

from configure_r2_bucket import lifecycle_rules
from r2_publish import FORECAST_KEEP_DAYS, MODEL_PRODUCTS, VERIFICATION_KEEP_DAYS


class ConfigureR2BucketTests(unittest.TestCase):
    def test_lifecycle_rules_have_one_day_cleanup_cushion(self):
        rules = lifecycle_rules()
        fallback = next(rule for rule in rules if rule["ID"] == "expire-all-model-data")
        self.assertEqual(fallback["Expiration"]["Days"], VERIFICATION_KEEP_DAYS + 1)

        forecast_rules = [rule for rule in rules if rule["ID"] != "expire-all-model-data"]
        self.assertEqual(len(forecast_rules), len(MODEL_PRODUCTS))
        self.assertTrue(
            all(rule["Expiration"]["Days"] == FORECAST_KEEP_DAYS + 1 for rule in forecast_rules)
        )


if __name__ == "__main__":
    unittest.main()
