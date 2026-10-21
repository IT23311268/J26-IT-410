"""Generate Stage 02 reports from persisted executed inspections; never auto-download."""

import sys

from ci_hgnn_rgd.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["data", "report", *sys.argv[1:]]))
