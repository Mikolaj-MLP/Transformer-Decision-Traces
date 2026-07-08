from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.logit_feature_score_suite.run_csqa_logit_feature_score_control_cap_sweep import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
