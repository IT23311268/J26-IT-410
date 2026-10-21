"""Run the existing offline Stage 03 CLI; all options are passed through unchanged."""

import sys

from ci_hgnn_rgd.cli import main as cli_main


def main(argv=None):
    return cli_main(["data", "stage03-run", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
