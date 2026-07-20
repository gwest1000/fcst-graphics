from __future__ import annotations

import unittest

import numpy as np

import fwi2025


class FWI2025ReferenceTest(unittest.TestCase):
    def test_first_petawawa_reference_hours(self) -> None:
        """Match the NRCan PRF2007 standard output distributed with cffdrs-ng."""

        weather = [
            # hour, temperature, RH, wind, rain
            (8, 14.78, 94.90, 2.7324, 0.0),
            (9, 17.44, 79.11, 3.6663, 0.0),
            (10, 21.21, 62.95, 4.3590, 0.0),
            (11, 23.60, 46.17, 8.2100, 0.0),
        ]
        expected = [
            # FFMC, DMC, DC, ISI, BUI, FWI
            (83.1265, 6.0167, 15.4092, 1.8814, 6.0893, 0.8842),
            (83.1165, 6.0976, 15.8583, 1.9695, 6.2180, 0.9346),
            (83.5392, 6.2721, 16.3639, 2.1545, 6.4059, 1.2034),
            (84.8979, 6.5541, 16.9054, 3.1394, 6.6565, 2.5077),
        ]
        state = fwi2025.FWI2025State.from_codes(np.array([85.0]), np.array([6.0]), np.array([15.0]))
        for values, reference in zip(weather, expected):
            hour, temp, rh, wind, rain = values
            output = fwi2025.step(
                state,
                np.array([temp]),
                np.array([rh]),
                np.array([wind]),
                np.array([rain]),
                np.array([True]),
                local_hour=float(hour),
            )
            actual = [output.ffmc[0], output.dmc[0], output.dc[0], output.isi[0], output.bui[0], output.fwi[0]]
            np.testing.assert_allclose(actual, reference, atol=6.0e-4, rtol=0.0)
            state = output.state

    def test_rain_intercept_is_cumulative_across_hours(self) -> None:
        state = fwi2025.FWI2025State.from_codes(np.array([85.0]), np.array([20.0]), np.array([100.0]))
        weather = (np.array([20.0]), np.array([40.0]), np.array([15.0]), np.array([False]))
        first = fwi2025.step(state, *weather[:3], np.array([0.4]), weather[3])
        second = fwi2025.step(first.state, *weather[:3], np.array([0.4]), weather[3])
        self.assertAlmostEqual(float(first.state.rain_total_mm[0]), 0.4, places=6)
        self.assertAlmostEqual(float(second.state.rain_total_mm[0]), 0.8, places=6)
        self.assertLess(float(second.ffmc[0]), float(first.ffmc[0]))

    def test_five_dry_hours_reset_rain_event(self) -> None:
        state = fwi2025.FWI2025State.from_codes(np.array([85.0]), np.array([20.0]), np.array([100.0]))
        state.rain_total_mm[:] = 2.0
        weather = (np.array([15.0]), np.array([60.0]), np.array([5.0]), np.array([0.0]), np.array([False]))
        for _ in range(5):
            output = fwi2025.step(state, *weather)
            state = output.state
        self.assertEqual(float(state.rain_total_mm[0]), 0.0)
        self.assertEqual(float(state.canopy_drying_hours[0]), 0.0)

    def test_reference_rain_event_updates_all_codes(self) -> None:
        """Match cffdrs-ng through interception, reset, and DMC threshold rain."""

        state = fwi2025.FWI2025State.from_codes(np.array([85.0]), np.array([6.0]), np.array([15.0]))
        output = None
        for precip in (0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.1):
            output = fwi2025.step(
                state,
                np.array([20.0]),
                np.array([40.0]),
                np.array([15.0]),
                np.array([precip]),
                np.array([True]),
            )
            state = output.state
        assert output is not None
        np.testing.assert_allclose(
            [output.ffmc[0], output.dmc[0], output.dc[0], output.isi[0], output.bui[0], output.fwi[0]],
            [74.762601, 7.723823, 19.3875, 1.609265, 7.739380, 0.849284],
            atol=6.0e-7,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
