"""Fail-fast validation for the repository's pinned Python dependencies.

The validator intentionally uses only the Python standard library so it can
report a malformed or incomplete environment without importing project
runtime dependencies first.  It validates both checked-in dependency roles,
then validates the selected installed environment (the test environment by
default), runs ``pip check``, and finally verifies production imports/exports.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON: Final[tuple[int, int]] = (3, 11)

RUNTIME_DIRECT_PATH: Final[Path] = Path("requirements/runtime-direct.txt")
TEST_DIRECT_PATH: Final[Path] = Path("requirements/test-direct.txt")
RUNTIME_LOCK_PATH: Final[Path] = Path("requirements.txt")
TEST_LOCK_PATH: Final[Path] = Path("requirements-test.txt")
PYTHON_VERSION_PATH: Final[Path] = Path(".python-version")

# Import name -> distribution name.  New third-party production imports must be
# added here and to runtime-direct.txt; unknown imports fail validation.
IMPORT_DISTRIBUTIONS: Final[Mapping[str, str]] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pydeck": "pydeck",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
}
REQUIRED_RUNTIME_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    IMPORT_DISTRIBUTIONS.values()
)
REQUIRED_TEST_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {"hypothesis", "pytest"}
)
BOOTSTRAP_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {"pip", "setuptools", "wheel"}
)
PRODUCTION_PACKAGES: Final[tuple[str, ...]] = ("app", "model", "planner")
# These planner modules are test, reporting, or ad-hoc analysis entry points;
# none is imported by planner's production public API or the Streamlit app.
NON_PRODUCTION_SOURCE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "planner/harness.py",
        "planner/properties.py",
        "planner/run_ci.py",
        "planner/run_export.py",
        "planner/run_model.py",
        "planner/run_real.py",
    }
)
REQUIRED_IMPORTS: Final[tuple[str, ...]] = (
    "streamlit",
    "pandas",
    "pyarrow",
    "pydeck",
    "numpy",
    "sklearn",
    "model",
    "planner",
)
REQUIRED_EXPORTS: Final[Mapping[str, tuple[str, ...]]] = {
    "model": ("fit_decline", "score", "is_safety_drop"),
    "planner": ("select", "summarize_selection", "run_backtest"),
}

_NAME_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_VERSION_PATTERN = r"[A-Za-z0-9][A-Za-z0-9.!+_-]*"
_EXACT_PIN_RE = re.compile(
    rf"^(?P<name>{_NAME_PATTERN})==(?P<version>{_VERSION_PATTERN})$"
)
_LOCK_PIN_RE = re.compile(
    rf"^(?P<pin>{_NAME_PATTERN}=={_VERSION_PATTERN})(?:\s+\\)?$"
)
_HASH_RE = re.compile(
    r"^--hash=sha256:(?P<digest>[0-9a-fA-F]{64})(?:\s+\\)?$"
)
_PYTHON_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")


class DependencyValidationError(RuntimeError):
    """A deterministic, non-sensitive dependency validation failure."""


@dataclass(frozen=True)
class RequirementPin:
    """One normalized exact dependency pin and its optional lock hashes."""

    name: str
    version: str
    hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DependencyContract:
    """Validated contents of all dependency declarations."""

    runtime_direct: Mapping[str, RequirementPin]
    test_direct: Mapping[str, RequirementPin]
    runtime_lock: Mapping[str, RequirementPin]
    test_lock: Mapping[str, RequirementPin]
    python_version: tuple[int, int]


def canonicalize_name(name: str) -> str:
    """Return the PEP 503 comparison form for a distribution name."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_label(path: Path, repository_root: Path) -> str:
    """Return only a repository-relative manifest label for diagnostics."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def _read_text(path: Path, repository_root: Path) -> str:
    label = _safe_label(path, repository_root)
    try:
        if not path.is_file():
            raise DependencyValidationError(f"required file is missing: {label}")
        return path.read_text(encoding="utf-8")
    except DependencyValidationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DependencyValidationError(
            f"required file is unreadable: {label} ({type(exc).__name__})"
        ) from None


def _parse_exact_pin(text: str, *, label: str, line_number: int) -> RequirementPin:
    match = _EXACT_PIN_RE.fullmatch(text)
    if match is None:
        raise DependencyValidationError(
            f"{label}:{line_number}: dependency entries must use exact name==version pins"
        )
    return RequirementPin(
        name=canonicalize_name(match.group("name")),
        version=match.group("version"),
    )


def parse_direct_manifest(
    path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, RequirementPin]:
    """Parse a direct manifest and reject every non-exact entry."""

    label = _safe_label(path, repository_root)
    pins: dict[str, RequirementPin] = {}
    for line_number, raw_line in enumerate(
        _read_text(path, repository_root).splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pin = _parse_exact_pin(line, label=label, line_number=line_number)
        if pin.name in pins:
            raise DependencyValidationError(
                f"{label}:{line_number}: duplicate dependency pin for {pin.name}"
            )
        pins[pin.name] = pin

    if not pins:
        raise DependencyValidationError(f"{label}: manifest contains no dependency pins")
    return pins


def parse_hashed_lock(
    path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, RequirementPin]:
    """Parse a pip-compatible lock and require exact pins with SHA-256 hashes."""

    label = _safe_label(path, repository_root)
    pins: dict[str, RequirementPin] = {}
    current: RequirementPin | None = None
    current_hashes: list[str] = []
    current_line = 0

    def finish_current() -> None:
        nonlocal current, current_hashes, current_line
        if current is None:
            return
        if not current_hashes:
            raise DependencyValidationError(
                f"{label}:{current_line}: resolved pin for {current.name} has no SHA-256 hash"
            )
        if current.name in pins:
            raise DependencyValidationError(
                f"{label}:{current_line}: duplicate resolved pin for {current.name}"
            )
        pins[current.name] = RequirementPin(
            current.name, current.version, tuple(current_hashes)
        )
        current = None
        current_hashes = []
        current_line = 0

    for line_number, raw_line in enumerate(
        _read_text(path, repository_root).splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        hash_match = _HASH_RE.fullmatch(line)
        if hash_match is not None:
            if current is None:
                raise DependencyValidationError(
                    f"{label}:{line_number}: hash is not attached to a resolved pin"
                )
            digest = hash_match.group("digest").lower()
            if digest in current_hashes:
                raise DependencyValidationError(
                    f"{label}:{line_number}: duplicate hash for {current.name}"
                )
            current_hashes.append(digest)
            continue

        lock_match = _LOCK_PIN_RE.fullmatch(line)
        if lock_match is not None and not raw_line[:1].isspace():
            finish_current()
            current = _parse_exact_pin(
                lock_match.group("pin"), label=label, line_number=line_number
            )
            current_line = line_number
            continue

        raise DependencyValidationError(
            f"{label}:{line_number}: lock entries must be exact pins or SHA-256 hashes"
        )

    finish_current()
    if not pins:
        raise DependencyValidationError(f"{label}: lock contains no resolved pins")
    return pins


def _read_python_version(
    path: Path, *, repository_root: Path
) -> tuple[int, int]:
    label = _safe_label(path, repository_root)
    lines = _read_text(path, repository_root).splitlines()
    if len(lines) != 1:
        raise DependencyValidationError(
            f"{label}: runtime declaration must contain one major.minor value"
        )
    match = _PYTHON_VERSION_RE.fullmatch(lines[0])
    if match is None:
        raise DependencyValidationError(
            f"{label}: runtime declaration must contain one major.minor value"
        )
    version = (int(match.group("major")), int(match.group("minor")))
    if version != SUPPORTED_PYTHON:
        supported = ".".join(str(part) for part in SUPPORTED_PYTHON)
        raise DependencyValidationError(
            f"{label}: supported Python version must be {supported}"
        )
    return version


def _pin_versions(pins: Mapping[str, RequirementPin]) -> dict[str, str]:
    return {name: pin.version for name, pin in pins.items()}


def _require_same_pin(
    direct: Mapping[str, RequirementPin],
    lock: Mapping[str, RequirementPin],
    *,
    direct_label: str,
    lock_label: str,
) -> None:
    for name in sorted(direct):
        if name not in lock:
            raise DependencyValidationError(
                f"{lock_label}: missing direct dependency {name} from {direct_label}"
            )
        if direct[name].version != lock[name].version:
            raise DependencyValidationError(
                f"{lock_label}: version for {name} does not match {direct_label}"
            )


def _production_local_import_names(repository_root: Path) -> set[str]:
    names = set(PRODUCTION_PACKAGES)
    for package_name in PRODUCTION_PACKAGES:
        package_path = repository_root / package_name
        if package_path.is_dir():
            names.update(path.stem for path in package_path.rglob("*.py"))
    names.update(path.stem for path in repository_root.glob("*.py"))
    return names


def discover_production_distributions(repository_root: Path) -> frozenset[str]:
    """Map every production third-party import to its distribution role."""

    local_names = _production_local_import_names(repository_root)
    standard_library = set(sys.stdlib_module_names) | {"__future__"}
    discovered: set[str] = set()

    for package_name in PRODUCTION_PACKAGES:
        package_path = repository_root / package_name
        if not package_path.is_dir():
            raise DependencyValidationError(
                f"production package is missing: {package_name}"
            )
        for source_path in sorted(package_path.rglob("*.py")):
            relative_source = source_path.relative_to(repository_root).as_posix()
            if relative_source in NON_PRODUCTION_SOURCE_PATHS:
                continue
            try:
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError) as exc:
                label = _safe_label(source_path, repository_root)
                raise DependencyValidationError(
                    f"cannot inspect production imports in {label} ({type(exc).__name__})"
                ) from None

            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])

            for imported_root in imported_roots:
                if imported_root in standard_library or imported_root in local_names:
                    continue
                distribution = IMPORT_DISTRIBUTIONS.get(imported_root)
                if distribution is None:
                    label = _safe_label(source_path, repository_root)
                    raise DependencyValidationError(
                        f"{label}: third-party import {imported_root} has no runtime distribution mapping"
                    )
                discovered.add(canonicalize_name(distribution))

    return frozenset(discovered)


def validate_manifest_contract(
    repository_root: Path = REPOSITORY_ROOT,
) -> DependencyContract:
    """Validate direct/lock syntax, role separation, and pin consistency."""

    root = repository_root.resolve()
    runtime_direct = parse_direct_manifest(
        root / RUNTIME_DIRECT_PATH, repository_root=root
    )
    test_direct = parse_direct_manifest(root / TEST_DIRECT_PATH, repository_root=root)
    runtime_lock = parse_hashed_lock(root / RUNTIME_LOCK_PATH, repository_root=root)
    test_lock = parse_hashed_lock(root / TEST_LOCK_PATH, repository_root=root)
    python_version = _read_python_version(
        root / PYTHON_VERSION_PATH, repository_root=root
    )

    runtime_names = frozenset(runtime_direct)
    test_names = frozenset(test_direct)
    discovered_runtime = discover_production_distributions(root)
    expected_runtime = REQUIRED_RUNTIME_DISTRIBUTIONS | discovered_runtime

    missing_runtime = sorted(expected_runtime - runtime_names)
    if missing_runtime:
        raise DependencyValidationError(
            "requirements/runtime-direct.txt: missing runtime dependency "
            + missing_runtime[0]
        )
    extra_runtime = sorted(runtime_names - expected_runtime)
    if extra_runtime:
        raise DependencyValidationError(
            "requirements/runtime-direct.txt: dependency has no production runtime role: "
            + extra_runtime[0]
        )

    overlap = sorted(runtime_names & test_names)
    if overlap:
        raise DependencyValidationError(
            f"direct dependency role overlap: {overlap[0]} appears in runtime and test manifests"
        )
    missing_test = sorted(REQUIRED_TEST_DISTRIBUTIONS - test_names)
    if missing_test:
        raise DependencyValidationError(
            "requirements/test-direct.txt: missing required test dependency "
            + missing_test[0]
        )
    if "pytest" in runtime_names:
        raise DependencyValidationError(
            "requirements/runtime-direct.txt: pytest is test-only"
        )

    _require_same_pin(
        runtime_direct,
        runtime_lock,
        direct_label="requirements/runtime-direct.txt",
        lock_label="requirements.txt",
    )
    _require_same_pin(
        runtime_direct,
        test_lock,
        direct_label="requirements/runtime-direct.txt",
        lock_label="requirements-test.txt",
    )
    _require_same_pin(
        test_direct,
        test_lock,
        direct_label="requirements/test-direct.txt",
        lock_label="requirements-test.txt",
    )

    leaked_test_packages = sorted(test_names & frozenset(runtime_lock))
    if leaked_test_packages:
        raise DependencyValidationError(
            "requirements.txt: test-only dependency is present: "
            + leaked_test_packages[0]
        )

    for name in sorted(runtime_lock):
        if name not in test_lock:
            raise DependencyValidationError(
                f"requirements-test.txt: runtime lock dependency is missing: {name}"
            )
        if runtime_lock[name].version != test_lock[name].version:
            raise DependencyValidationError(
                f"requirements-test.txt: runtime dependency version drift for {name}"
            )

    return DependencyContract(
        runtime_direct=runtime_direct,
        test_direct=test_direct,
        runtime_lock=runtime_lock,
        test_lock=test_lock,
        python_version=python_version,
    )


def validate_current_interpreter(
    expected: tuple[int, int],
    *,
    version_info: Sequence[int] = sys.version_info,
) -> None:
    """Require the interpreter major/minor declared by the repository."""

    actual = (int(version_info[0]), int(version_info[1]))
    if actual != expected:
        expected_text = ".".join(str(part) for part in expected)
        actual_text = ".".join(str(part) for part in actual)
        raise DependencyValidationError(
            f"Python version mismatch: expected {expected_text}, installed {actual_text}"
        )


def get_installed_versions() -> dict[str, str]:
    """Return one exact metadata version for each installed distribution."""

    installed: dict[str, str] = {}
    try:
        distributions: Iterable[importlib.metadata.Distribution] = (
            importlib.metadata.distributions()
        )
        for distribution in distributions:
            raw_name = distribution.metadata.get("Name")
            version = distribution.version
            if not raw_name or not version:
                raise DependencyValidationError(
                    "installed distribution has incomplete package metadata"
                )
            name = canonicalize_name(raw_name)
            prior = installed.get(name)
            if prior is not None and prior != version:
                raise DependencyValidationError(
                    f"multiple installed versions detected for {name}"
                )
            installed[name] = version
    except DependencyValidationError:
        raise
    except Exception as exc:
        raise DependencyValidationError(
            f"installed package metadata is unavailable ({type(exc).__name__})"
        ) from None
    return installed


def validate_installed_versions(
    lock: Mapping[str, RequirementPin],
    *,
    installed: Mapping[str, str] | None = None,
) -> None:
    """Require a complete exact lock representation of the environment."""

    actual = (
        {canonicalize_name(name): version for name, version in installed.items()}
        if installed is not None
        else get_installed_versions()
    )

    for name in sorted(lock):
        if name not in actual:
            raise DependencyValidationError(
                f"locked distribution is not installed: {name}=={lock[name].version}"
            )
        if actual[name] != lock[name].version:
            raise DependencyValidationError(
                f"installed version mismatch for {name}: expected {lock[name].version}, installed {actual[name]}"
            )

    unlocked = sorted(set(actual) - set(lock) - BOOTSTRAP_DISTRIBUTIONS)
    if unlocked:
        raise DependencyValidationError(
            f"installed distribution is not represented in the selected lock: {unlocked[0]}"
        )


def validate_dependency_graph(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Run pip's installed dependency compatibility check without a network call."""

    command = [sys.executable, "-m", "pip", "check"]
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DependencyValidationError(
            f"installed dependency graph check could not run ({type(exc).__name__})"
        ) from None
    if result.returncode != 0:
        raise DependencyValidationError(
            "installed dependency graph is incompatible; run 'python -m pip check' for package details"
        )


def _require_repository_module(
    module: ModuleType, package_name: str, repository_root: Path
) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise DependencyValidationError(
            f"{package_name} did not resolve to a repository package"
        )
    try:
        resolved = Path(module_file).resolve()
        expected_root = (repository_root / package_name).resolve()
        if not resolved.is_relative_to(expected_root):
            raise DependencyValidationError(
                f"{package_name} did not resolve to the repository's real package"
            )
    except OSError as exc:
        raise DependencyValidationError(
            f"{package_name} package location is unreadable ({type(exc).__name__})"
        ) from None


def _require_external_module(
    module: ModuleType, module_name: str, repository_root: Path
) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return
    try:
        if Path(module_file).resolve().is_relative_to(repository_root.resolve()):
            raise DependencyValidationError(
                f"required import {module_name} is shadowed by repository source"
            )
    except OSError as exc:
        raise DependencyValidationError(
            f"required import {module_name} has an unreadable location ({type(exc).__name__})"
        ) from None


def validate_runtime_imports(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> None:
    """Import required runtime modules and verify real project API exports."""

    root = repository_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    imported: dict[str, ModuleType] = {}
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importer(module_name)
        except Exception as exc:
            # Deliberately omit exception text: import exceptions may contain local
            # paths, environment values, or other deployment-sensitive details.
            raise DependencyValidationError(
                f"required import failed: {module_name} ({type(exc).__name__})"
            ) from None
        imported[module_name] = module
        if module_name in PRODUCTION_PACKAGES:
            _require_repository_module(module, module_name, root)
        else:
            _require_external_module(module, module_name, root)

    for package_name, export_names in REQUIRED_EXPORTS.items():
        module = imported[package_name]
        for export_name in export_names:
            if not callable(getattr(module, export_name, None)):
                raise DependencyValidationError(
                    f"{package_name} package is missing callable export {export_name}"
                )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pinned manifests, installed packages, graph, and imports."
    )
    parser.add_argument(
        "--scope",
        choices=("runtime", "test"),
        default="test",
        help="installed environment to validate (default: test)",
    )
    return parser


def run_validation(scope: str, *, repository_root: Path = REPOSITORY_ROOT) -> int:
    """Run the complete fail-fast gate and return its selected lock size."""

    contract = validate_manifest_contract(repository_root)
    validate_current_interpreter(contract.python_version)
    selected_lock = (
        contract.runtime_lock if scope == "runtime" else contract.test_lock
    )
    validate_installed_versions(selected_lock)
    validate_dependency_graph()
    validate_runtime_imports(repository_root=repository_root)
    return len(selected_lock)


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        distribution_count = run_validation(args.scope)
    except DependencyValidationError as exc:
        print(f"Dependency validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Keep unexpected failures non-sensitive while preserving a useful class.
        print(
            f"Dependency validation failed: internal validator error ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2

    print(
        f"Dependency validation passed for {args.scope} scope "
        f"({distribution_count} locked distributions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
