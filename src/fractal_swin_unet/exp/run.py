"""CLI entrypoint for running experiment matrices."""

from __future__ import annotations

import argparse

from .runner import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix YAML.")
    args = parser.parse_args()

    run_matrix(args.matrix)


if __name__ == "__main__":
    main()
