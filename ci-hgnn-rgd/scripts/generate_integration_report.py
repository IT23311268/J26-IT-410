"""Republish reports from an existing complete Stage 03 run; never reacquire data."""

import sys

from ci_hgnn_rgd.cli import main as cli_main


def main(argv=None):
    return cli_main(["data", "integration-report", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
