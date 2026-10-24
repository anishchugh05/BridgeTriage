"""Focused tests for the fail-fast dependency validation gate."""

from __future__ import annotations

import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from scripts import validate_dependencies as dependencies


_HASH = "a" * 64
_RUNTIME_PINS = {
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "pyarrow": "26.0.0",
    "pydeck": "0.9.3",
    "scikit-learn": "1.9.1",
    "streamlit": "1.66.1",
}
_TEST_PINS = {"hypothesis": "6.170.9", "pytest": "9.1.1"}


def _direct_text(pins: dict[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in pins.items())


def _lock_text(pins: dict[str, str]) -> str:
    return "".join(
        f"{name}=={version} \\\n    --hash=sha256:{_HASH}\n" for name, version in pins.items()
    )


class DependencyManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "requirements").mkdir()
        for package in ("app", "model", "planner"):
            (self.root / package).mkdir()
            (self.root / package / "__init__.py").write_text("", encoding="utf-8")

        # Exercise static discovery for every direct import except PyArrow, which
        # is an explicitly required pandas parquet engine/runtime import check.
        (self.root / "app" / "imports.py").write_text(
            "import pandas\nimport pydeck\nimport streamlit\n", encoding="utf-8"
        )
        (self.root / "model" / "imports.py").write_text(
            "import numpy\nfrom sklearn.linear_model import LogisticRegression\n",
            encoding="utf-8",
        )
        (self.root / ".python-version").write_text("3.11\n", encoding="utf-8")
        self._write_manifests()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifests(self) -> None:
        (self.root / "requirements" / "runtime-direct.txt").write_text(
            _direct_text(_RUNTIME_PINS), encoding="utf-8"
        )
        (self.root / "requirements" / "test-direct.txt").write_text(
            _direct_text(_TEST_PINS), encoding="utf-8"
        )
        (self.root / "requirements.txt").write_text(
            _lock_text(_RUNTIME_PINS), encoding="utf-8"
        )
        (self.root / "requirements-test.txt").write_text(
            _lock_text(_RUNTIME_PINS | _TEST_PINS), encoding="utf-8"
        )

    def test_accepts_exact_separate_hashed_manifests(self) -> None:
        contract = dependencies.validate_manifest_contract(self.root)

        self.assertEqual(set(contract.runtime_direct), set(_RUNTIME_PINS))
        self.assertEqual(set(contract.test_direct), set(_TEST_PINS))
        self.assertEqual(contract.python_version, (3, 11))

    def test_rejects_non_exact_direct_entry(self) -> None:
        path = self.root / "requirements" / "runtime-direct.txt"
        path.write_text("numpy>=2.3.5\n", encoding="utf-8")

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "exact name==version"
        ):
            dependencies.validate_manifest_contract(self.root)

    def test_rejects_resolved_pin_without_hash(self) -> None:
        path = self.root / "requirements.txt"
        path.write_text("numpy==2.3.5\n", encoding="utf-8")

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "has no SHA-256 hash"
        ):
            dependencies.validate_manifest_contract(self.root)

    def test_rejects_direct_role_overlap(self) -> None:
        path = self.root / "requirements" / "test-direct.txt"
        path.write_text(
            _direct_text(_TEST_PINS | {"pandas": "2.3.3"}), encoding="utf-8"
        )

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "role overlap"
        ):
            dependencies.validate_manifest_contract(self.root)

    def test_rejects_direct_and_lock_version_drift(self) -> None:
        drifted = dict(_RUNTIME_PINS)
        drifted["numpy"] = "2.3.4"
        (self.root / "requirements.txt").write_text(
            _lock_text(drifted), encoding="utf-8"
        )

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "does not match"
        ):
            dependencies.validate_manifest_contract(self.root)


class InstalledEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = {
            name: dependencies.RequirementPin(name, version, (_HASH,))
            for name, version in (_RUNTIME_PINS | _TEST_PINS).items()
        }
        self.installed = dict(_RUNTIME_PINS | _TEST_PINS)

    def test_accepts_exact_lock_environment_and_bootstrap_tools(self) -> None:
        dependencies.validate_installed_versions(
            self.lock, installed=self.installed | {"pip": "24.0"}
        )

    def test_rejects_missing_mismatched_and_unrepresented_distributions(self) -> None:
        missing = dict(self.installed)
        missing.pop("numpy")
        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "not installed: numpy"
        ):
            dependencies.validate_installed_versions(self.lock, installed=missing)

        mismatched = self.installed | {"numpy": "0.0.1"}
        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "version mismatch for numpy"
        ):
            dependencies.validate_installed_versions(self.lock, installed=mismatched)

        unlocked = self.installed | {"unlocked-package": "1.0"}
        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "not represented"
        ):
            dependencies.validate_installed_versions(self.lock, installed=unlocked)

    def test_rejects_an_incompatible_installed_graph(self) -> None:
        def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 1, "private output", "private error")

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError, "graph is incompatible"
        ) as caught:
            dependencies.validate_dependency_graph(runner=runner)

        self.assertNotIn("private", str(caught.exception))


class RuntimeImportTests(unittest.TestCase):
    def test_import_failure_does_not_expose_exception_details(self) -> None:
        def importer(module_name: str) -> types.ModuleType:
            raise RuntimeError(r"C:\\secret\\path token=do-not-print")

        with self.assertRaisesRegex(
            dependencies.DependencyValidationError,
            r"required import failed: streamlit \(RuntimeError\)",
        ) as caught:
            dependencies.validate_runtime_imports(importer=importer)

        message = str(caught.exception)
        self.assertNotIn("secret", message)
        self.assertNotIn("token", message)

    def test_rejects_missing_real_backtest_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modules: dict[str, types.ModuleType] = {}
            for module_name in dependencies.REQUIRED_IMPORTS:
                module = types.ModuleType(module_name)
                if module_name in ("model", "planner"):
                    package_file = root / module_name / "__init__.py"
                    package_file.parent.mkdir()
                    package_file.write_text("", encoding="utf-8")
                    module.__file__ = str(package_file)
                else:
                    module.__file__ = str(root.parent / f"{module_name}.py")
                modules[module_name] = module

            modules["model"].fit_decline = lambda: None
            modules["model"].score = lambda: None
            modules["model"].is_safety_drop = lambda: None
            modules["planner"].select = lambda: None
            modules["planner"].summarize_selection = lambda: None

            with self.assertRaisesRegex(
                dependencies.DependencyValidationError,
                "planner package is missing callable export run_backtest",
            ):
                dependencies.validate_runtime_imports(
                    repository_root=root, importer=modules.__getitem__
                )


if __name__ == "__main__":
    unittest.main()
