import sys
from pathlib import Path

ADDON_SRC = Path(__file__).resolve().parents[1] / "addon_src"
if str(ADDON_SRC) not in sys.path:
    sys.path.insert(0, str(ADDON_SRC))

# The RWKV integration harness must set the vendor root before any
# rwkvp_addon module is imported, so do it once for the whole session.
_INTEGRATION = Path(__file__).resolve().parent / "integration"
if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))
try:
    import rwkv_harness

    if rwkv_harness.unavailable_reason() is None:
        rwkv_harness.prepare_environment()
except Exception:  # integration deps missing: those tests skip themselves
    pass
