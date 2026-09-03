from __future__ import annotations

import unittest

import matplotlib.colors as mcolors
import numpy as np

import make_hrdps_west_convective as hrdps
import make_hrdps_west_fourpanel as fourpanel
import plot_style


class HrdpsFourPanelTest(unittest.TestCase):
    def tearDown(self) -> None:
        fourpanel.set_model("west")

    def test_transmission_lines_are_limited_to_right_hand_panels(self) -> None:
        self.assertEqual(fourpanel.TRANSMISSION_PANEL_INDICES, (1, 3))

    def test_continental_domain_reaches_the_bc_yukon_border_and_preserves_aspect(self) -> None:
        fourpanel.set_model("continental")
        expanded = fourpanel.fourpanel_extent()
        original = fourpanel.model_config().extent

        self.assertEqual(expanded, (-139.6, -105.3, 45.5, 59.8))
        self.assertLessEqual(expanded[2], 45.5)
        self.assertGreaterEqual(expanded[3], 59.7)
        self.assertLess(expanded[0], original[0])
        self.assertGreater(expanded[1], original[1])

        def projected_ratio(extent: tuple[float, float, float, float]) -> float:
            west, east, south, north = extent
            edge_points = np.linspace(0.0, 1.0, 101)
            longitude = np.concatenate(
                (
                    west + (east - west) * edge_points,
                    west + (east - west) * edge_points,
                    np.full_like(edge_points, west),
                    np.full_like(edge_points, east),
                )
            )
            latitude = np.concatenate(
                (
                    np.full_like(edge_points, south),
                    np.full_like(edge_points, north),
                    south + (north - south) * edge_points,
                    south + (north - south) * edge_points,
                )
            )
            points = fourpanel.PANEL_PROJ.transform_points(fourpanel.DATA_CRS, longitude, latitude)
            return np.ptp(points[:, 0]) / np.ptp(points[:, 1])

        self.assertAlmostEqual(projected_ratio(expanded), projected_ratio(original), delta=0.01)

    def test_static_model_topography_is_required_at_its_available_hour(self) -> None:
        stamp = "20260720T06Z"
        names = fourpanel.required_names(stamp, hrdps.TERRAIN_FHOUR)

        self.assertIn(
            fourpanel.field_name("HGT", "SFC", "0", stamp, hrdps.TERRAIN_FHOUR),
            names,
        )

    def test_mslp_smoothing_is_large_enough_to_suppress_grid_scale_noise(self) -> None:
        self.assertGreaterEqual(fourpanel.MSLP_SMOOTHING_KM, 10.0)

    def test_all_mslp_levels_above_102_4_are_in_blue_group(self) -> None:
        minor, major, threshold, high = fourpanel.mslp_contour_groups()
        combined = np.sort(np.concatenate((minor, major, threshold, high)))

        np.testing.assert_array_equal(combined, fourpanel.MSLP_LEVELS_KPA)
        np.testing.assert_array_equal(
            high,
            fourpanel.MSLP_LEVELS_KPA[fourpanel.MSLP_LEVELS_KPA > 102.4],
        )
        np.testing.assert_array_equal(threshold, [102.4])
        self.assertTrue(np.all(np.concatenate((minor, major)) < 102.4))

    def test_contour_smoothing_and_height_emphasis_are_physical_scale(self) -> None:
        self.assertGreaterEqual(fourpanel.TEMP850_SMOOTHING_KM, 6.0)
        self.assertGreaterEqual(fourpanel.IPW_SMOOTHING_KM, 7.0)
        self.assertGreaterEqual(fourpanel.LI_SMOOTHING_KM, 7.0)
        self.assertGreaterEqual(fourpanel.CAPE_SMOOTHING_KM, 8.0)
        self.assertTrue(all(width >= 1.75 for width in fourpanel.LI_LINEWIDTHS))
        self.assertTrue(all(left > right for left, right in zip(fourpanel.LI_LINEWIDTHS, fourpanel.LI_LINEWIDTHS[1:])))
        self.assertGreater(fourpanel.HGT500_LINEWIDTH, 1.25)
        self.assertGreater(fourpanel.HGT500_HALO_LINEWIDTH, fourpanel.HGT500_LINEWIDTH)
        self.assertEqual(fourpanel.HGT500_INTERVAL_KM, 0.06)
        self.assertEqual(fourpanel.HGT500_ANCHOR_KM, 6.00)
        np.testing.assert_allclose(np.diff(fourpanel.HGT500_LEVELS_KM), fourpanel.HGT500_INTERVAL_KM)
        self.assertIn(5.76, fourpanel.HGT500_LEVELS_KM)
        self.assertIn(5.82, fourpanel.HGT500_LEVELS_KM)
        self.assertIn(6.00, fourpanel.HGT500_LEVELS_KM)
        self.assertEqual(fourpanel.HGT500_LABEL_FORMAT % 6.00, "6.00")

    def test_850_temperature_style_groups_are_exclusive_and_complete(self) -> None:
        standard, zero, warm, hot = fourpanel.temp850_contour_groups()
        combined = np.sort(np.concatenate((standard, zero, warm, hot)))

        np.testing.assert_array_equal(combined, fourpanel.TEMP850_LEVELS_C)
        np.testing.assert_array_equal(zero, [0])
        np.testing.assert_array_equal(warm, [16, 18])
        self.assertTrue(np.all(hot >= 20))
        self.assertGreater(fourpanel.TEMP850_STANDARD_LINEWIDTH, 1.05)

    def test_fourpanel_colorbars_fill_plot_height_and_reach_right_border(self) -> None:
        backdrop = plot_style.FOURPANEL_COLORBAR_BACKDROP
        colorbar = plot_style.FOURPANEL_COLORBAR_AX
        plot_height = 1.0 - plot_style.FOURPANEL_HEADER_BAND_HEIGHT - plot_style.FOURPANEL_FOOTER_BAND_HEIGHT

        self.assertAlmostEqual(backdrop[0] + backdrop[2], 1.0)
        self.assertAlmostEqual(colorbar[0] + colorbar[2], 1.0)
        self.assertAlmostEqual(colorbar[1], plot_style.FOURPANEL_FOOTER_BAND_HEIGHT)
        self.assertAlmostEqual(colorbar[3], plot_height)

    def test_precip_colorbar_is_slightly_inset_from_text_bands(self) -> None:
        colorbar = plot_style.FOURPANEL_PRECIP_COLORBAR_AX
        plot_top = 1.0 - plot_style.FOURPANEL_HEADER_BAND_HEIGHT

        self.assertAlmostEqual(colorbar[0] + colorbar[2], 1.0)
        self.assertGreater(colorbar[1], plot_style.FOURPANEL_FOOTER_BAND_HEIGHT)
        self.assertLess(colorbar[1] + colorbar[3], plot_top)

    def test_terrain_palette_darkens_with_elevation(self) -> None:
        luminance = []
        for color in fourpanel.TERRAIN_COLORS:
            red, green, blue = mcolors.to_rgb(color)
            luminance.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)

        self.assertTrue(np.all(np.diff(luminance) < 0.0))

    def test_ipw_palette_matches_operational_scale_interval_by_interval(self) -> None:
        cmap, norm, levels = fourpanel.make_ipw_cmap()

        np.testing.assert_array_equal(levels, np.arange(10, 52, 2))
        self.assertEqual([mcolors.to_hex(color) for color in cmap.colors], fourpanel.IPW_COLORS)
        self.assertEqual(mcolors.to_hex(cmap.get_under()), fourpanel.IPW_UNDER_COLOR)
        self.assertEqual(mcolors.to_hex(cmap.get_over()), fourpanel.IPW_OVER_COLOR)
        self.assertEqual(norm.N, len(levels))

    def test_short_period_precip_palette_matches_operational_scale_interval_by_interval(self) -> None:
        cmap, norm, levels = fourpanel.make_precip_cmap()

        self.assertEqual(levels, fourpanel.PRECIP_LEVELS_MM)
        self.assertEqual([mcolors.to_hex(color) for color in cmap.colors], fourpanel.PRECIP_COLORS)
        self.assertEqual(mcolors.to_hex(cmap.get_over()), fourpanel.PRECIP_OVER_COLOR)
        self.assertEqual(norm.N, len(levels))
        self.assertEqual(
            fourpanel.PRECIP_TICKS_MM,
            [0.25, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 60, 80, 100],
        )


if __name__ == "__main__":
    unittest.main()
