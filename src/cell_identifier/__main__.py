"""Allow ``python -m cell_identifier``."""

import sys

from .cli import main

sys.exit(main())
