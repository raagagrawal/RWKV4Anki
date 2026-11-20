import sys
from pathlib import Path

ADDON_SRC = Path(__file__).resolve().parents[1] / "addon_src"
if str(ADDON_SRC) not in sys.path:
    sys.path.insert(0, str(ADDON_SRC))
