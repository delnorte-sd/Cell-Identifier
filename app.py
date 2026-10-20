"""WSGI entry point kept at the repository root for ``gunicorn app:app`` (Render).

The application itself lives in :mod:`cell_identifier.web`. For local use prefer
``cell-identifier serve``.
"""

import sys
from pathlib import Path

try:
    from cell_identifier.web import create_app
except ImportError:  # running from a source checkout without `pip install`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from cell_identifier.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
