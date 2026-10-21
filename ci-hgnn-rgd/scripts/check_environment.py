"""Convenience entry point; install the package before invoking this script."""

import sys

from ci_hgnn_rgd.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-env", *sys.argv[1:]]))
