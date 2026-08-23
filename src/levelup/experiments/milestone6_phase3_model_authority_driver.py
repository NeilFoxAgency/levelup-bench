"""CLI for publishing the completed Phase 3 model-artifact authority."""

from __future__ import annotations

import argparse
from pathlib import Path

from levelup.experiments.milestone6_phase3_model_authority import (
    build_phase3_model_artifact_authority,
    canonical_phase3_model_authority_bytes,
    write_phase3_model_authority,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authority-repository", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)
    authority = build_phase3_model_artifact_authority(
        args.output_root,
        authority_repository=args.authority_repository,
    )
    payload = canonical_phase3_model_authority_bytes(authority)
    if args.output_path is not None:
        write_phase3_model_authority(args.output_path, authority)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
