import sys
from pathlib import Path

# Allow `import schema`, `import ingestion`, `import api` when running
# pytest from the repo root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
