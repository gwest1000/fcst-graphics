from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import project_paths


class ProjectPathsTests(unittest.TestCase):
    def test_local_fallback_uses_repository_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            self.assertEqual(
                project_paths.data_root({}, project_root),
                project_root / "data",
            )
            self.assertEqual(
                project_paths.plot_root({}, project_root),
                project_root / "plots",
            )

    def test_shared_root_namespaces_each_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            data = shared / "fcstGraphics" / "data"
            plots = shared / "fcstGraphics" / "plots"
            data.mkdir(parents=True)
            plots.mkdir()
            env = {"PROJECT_DATA_ROOT": str(shared)}
            self.assertEqual(project_paths.data_root(env), data.resolve())
            self.assertEqual(project_paths.plot_root(env), plots.resolve())

    def test_project_override_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            override = root / "override"
            shared.mkdir()
            override.mkdir()
            env = {
                "PROJECT_DATA_ROOT": str(shared),
                "FCSTGRAPHICS_DATA_ROOT": str(override),
            }
            self.assertEqual(project_paths.data_root(env), override.resolve())

    def test_configured_missing_root_fails_instead_of_using_internal_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-mounted"
            with self.assertRaisesRegex(RuntimeError, "configured but unavailable"):
                project_paths.data_root({"PROJECT_DATA_ROOT": str(missing)})

    def test_machine_config_supplies_shared_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "project-data"
            (shared / "fcstGraphics" / "data").mkdir(parents=True)
            (shared / "fcstGraphics" / "plots").mkdir()
            config = Path(tmp) / "project-data.env"
            config.write_text(f"PROJECT_DATA_ROOT={shared}\n")
            with mock.patch.dict(
                project_paths.os.environ,
                {"PROJECT_DATA_CONFIG": str(config)},
                clear=True,
            ):
                self.assertEqual(
                    project_paths.data_root(),
                    (shared / "fcstGraphics" / "data").resolve(),
                )


if __name__ == "__main__":
    unittest.main()
