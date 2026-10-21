"""Fetch bounded official documentation/source metadata; never execute fetched code."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    root = Path(__file__).resolve().parents[1] / "data/raw/source_audits"
    root.mkdir(parents=True, exist_ok=True)
    observations = []

    def fetch(name, url, head=False):
        try:
            with urlopen(
                Request(
                    url,
                    method="HEAD" if head else "GET",
                    headers={"User-Agent": "CI-HGNN-RGD-Stage02-source-audit/0.1"},
                ),
                timeout=30,
            ) as response:
                data = b"" if head else response.read(2_000_001)
                if len(data) > 2_000_000:
                    raise ValueError("source metadata exceeds 2 MB audit limit")
                record = dict(
                    name=name,
                    url=url,
                    final_url=response.url,
                    retrieved_at=datetime.now(UTC).isoformat(),
                    status=response.status,
                    content_length=response.headers.get("Content-Length"),
                    content_type=response.headers.get("Content-Type"),
                    etag=response.headers.get("ETag"),
                    sha256=hashlib.sha256(data).hexdigest() if not head else None,
                )
                observations.append(record)
                if not head:
                    destination = root / name
                    if not destination.exists():
                        destination.write_bytes(data)
                print(json.dumps(record))
                return data
        except Exception as error:
            record = dict(name=name, url=url, error=f"{type(error).__name__}: {error}")
            observations.append(record)
            print(json.dumps(record))
            return None

    fetch("scicite-readme.md", "https://raw.githubusercontent.com/allenai/scicite/master/README.md")
    fetch("scicite-license.txt", "https://raw.githubusercontent.com/allenai/scicite/master/LICENSE")
    fetch(
        "scicite-head",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz",
        True,
    )
    fetch(
        "ogb-dataset.py.txt",
        "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/dataset.py",
    )
    fetch(
        "ogb-master.csv",
        "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv",
    )
    repo = fetch("h2gb-repository.json", "https://api.github.com/repos/SpaceLearner/H2GB")
    repository = "SpaceLearner/H2GB"
    if repo is None:
        repository = "junhongmit/H2GB"
        repo = fetch(
            "h2gb-alternative-repository.json", "https://api.github.com/repos/junhongmit/H2GB"
        )
    if repo:
        metadata = json.loads(repo)
        branch = metadata["default_branch"]
        tree = fetch(
            "h2gb-tree.json",
            f"https://api.github.com/repos/{repository}/git/trees/{branch}?recursive=1",
        )
        if tree:
            paths = [
                item["path"]
                for item in json.loads(tree).get("tree", [])
                if item["path"].endswith(".py") and "mag" in item["path"].lower()
            ]
            print("H2GB MAG source paths:", paths)
            for index, path in enumerate(paths[:5]):
                fetch(
                    f"h2gb-mag-source-{index}.py.txt",
                    f"https://raw.githubusercontent.com/{repository}/{branch}/{path}",
                )
    fetch(
        "kaggle-metadata.json",
        "https://www.kaggle.com/api/v1/datasets/list/Cornell-University/arxiv",
    )
    target = root / ("preflight-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + ".json")
    target.write_text(json.dumps(observations, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
