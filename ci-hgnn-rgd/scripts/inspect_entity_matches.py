"""Trace retained records, match decisions and provenance using --id and --run-dir."""

import sys

from ci_hgnn_rgd.cli import main as cli_main


def main(argv=None):
    return cli_main(["data", "trace-record", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
