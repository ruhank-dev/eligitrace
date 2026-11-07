"""Protects the held-out test split: it may be evaluated only through the one-time final run."""

from settings.config import get_settings


class TestSplitLocked(RuntimeError):
    pass


def final_dir():
    return get_settings().results_dir / "final_test"


def check_split_allowed(split: str, final: bool) -> None:
    if split != "test":
        return
    if not final:
        raise TestSplitLocked(
            "The test split is locked. All tuning uses 'val'. "
            "Run the one-time frozen evaluation with `task eval-final`."
        )
    if not (final_dir() / "FROZEN.json").exists():
        raise TestSplitLocked("Write results/final_test/FROZEN.json (via evaluation.final) before testing.")
