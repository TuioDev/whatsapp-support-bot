"""Vercel entry point: the platform serves the ASGI app exported here.

vercel.json rewrites every path to this function and the original request path
is preserved, so the routes declared in app.py answer at their own paths
(/webhook, /health). All the logic lives in app.py; this file only re-exports.
"""

import sys
from pathlib import Path

# The function runs from api/, so make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402

__all__ = ["app"]
