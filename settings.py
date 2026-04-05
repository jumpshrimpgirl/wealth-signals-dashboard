"""Environment-backed settings for optional integrations."""

import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
