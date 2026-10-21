"""Inspect configured local sources independently; a blocked source does not stop others."""

import argparse
import json

from ci_hgnn_rgd.config import load_config
from ci_hgnn_rgd.data.service import DataService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-records", type=int, default=1000)
    args = parser.parse_args()
    service = DataService(load_config(args.config))
    results = {}
    for dataset in ("scicite", "arxiv", "mag_year", "ogbn_arxiv"):
        try:
            report = service.inspect(dataset, max_records=args.max_records, offline=True)
            results[dataset] = {"status": report.get("status", "INSPECTED")}
        except (OSError, ValueError, ImportError) as error:
            results[dataset] = {"blocked_or_failed": str(error)}
    print(json.dumps(results, indent=2))
    return int(any("blocked_or_failed" in value for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
