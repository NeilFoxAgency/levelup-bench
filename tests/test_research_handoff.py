from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RESEARCH_DOCS = (
    "AGENTS.md",
    "docs/README.md",
    "docs/research-vision.md",
    "docs/research-history.md",
    "docs/benchmark-contract.md",
    "docs/research-methodology.md",
    "docs/milestone-6-research-plan.md",
    "docs/metrics-and-reporting.md",
    "docs/prior-art-and-reuse.md",
    "docs/speedrun-tas-roadmap.md",
    "docs/compute-and-reproducibility.md",
    "docs/future-research-agenda.md",
    "experiments/README.md",
)

HISTORICAL_REFERENCE_ARTIFACTS = (
    "experiments/milestone3_reference.json",
    "experiments/milestone4_reference.json",
    "experiments/milestone5_reference.json",
)


def test_research_handoff_documents_exist() -> None:
    missing = [path for path in REQUIRED_RESEARCH_DOCS if not (ROOT / path).is_file()]
    assert not missing, f"missing research handoff documents: {missing}"


def test_agents_reading_map_references_every_core_document() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for path in REQUIRED_RESEARCH_DOCS[1:-1]:
        assert path in agents or path == "docs/README.md", f"AGENTS.md does not reference {path}"


def test_historical_reference_artifacts_are_preserved() -> None:
    missing = [path for path in HISTORICAL_REFERENCE_ARTIFACTS if not (ROOT / path).is_file()]
    assert not missing, f"missing historical reference artifacts: {missing}"


def test_private_game_and_run_directories_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in ("runs/", "scratch/", "roms/", "data/private/", "emulator_states/"):
        assert entry in gitignore, f"expected {entry} to be ignored"
