"""Allow ``python -m glider_dashboard``."""

import sys

from .cli import main

sys.exit(main())
