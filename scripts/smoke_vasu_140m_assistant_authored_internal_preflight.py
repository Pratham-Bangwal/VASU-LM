"""Run the read-only VASU-140M assistant-authored suite preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.framework.vasu_140m_assistant_authored_internal_preflight import (
        qualify_internal_suite,
    )

    print(json.dumps(qualify_internal_suite(ROOT), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
