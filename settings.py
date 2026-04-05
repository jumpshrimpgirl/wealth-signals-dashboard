"""Environment-backed settings for optional integrations."""

import os

# Set WEALTH_SIGNALS_SHOW_DEBUG=1 to expose debug_signal_reasons / debug_match_reasons in pipeline rows.
SHOW_DEBUG = os.environ.get("WEALTH_SIGNALS_SHOW_DEBUG", "").lower() in ("1", "true", "yes")
