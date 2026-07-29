"""Print a deterministic, read-only VASU artifact lineage index."""

from __future__ import annotations

import json
from pathlib import Path

from vasu.utils.lineage import build_lineage_index


if __name__ == "__main__":
    print(json.dumps(build_lineage_index(Path.cwd()), indent=2, sort_keys=True))
