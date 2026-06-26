import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from deep_cfr_poker.plotting import format_chart_title  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPO_ROOT / "deep_cfr_poker",
    REPO_ROOT / "experiments",
)


def test_chart_title_prefix_includes_algorithm_and_poker_variant():
    assert (
        format_chart_title(
            "Exploitability",
            algorithm_variant="ESCHER",
            game_name="kuhn_poker",
        )
        == "ESCHER - Kuhn - Exploitability"
    )
    assert (
        format_chart_title(
            "Average Policy Value",
            algorithm_variant="Deep CFR",
            game_name="leduc_poker",
        )
        == "Deep CFR - Leduc - Average Policy Value"
    )


def test_chart_title_prefix_is_not_duplicated():
    assert (
        format_chart_title(
            "ESCHER - Kuhn - Exploitability",
            algorithm_variant="ESCHER",
            game_name="kuhn_poker",
        )
        == "ESCHER - Kuhn - Exploitability"
    )


def test_chart_titles_are_routed_through_shared_helper():
    forbidden_tokens = (".set_title(", ".suptitle(", "plt.title(")
    direct_title_calls = []

    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if not any(token in line for token in forbidden_tokens):
                    continue
                stripped = line.strip()
                if (
                    relative_path == "deep_cfr_poker/plotting.py"
                    and stripped.startswith("ax.set_title(")
                ):
                    continue
                direct_title_calls.append(f"{relative_path}:{line_number}: {stripped}")

    assert direct_title_calls == []
