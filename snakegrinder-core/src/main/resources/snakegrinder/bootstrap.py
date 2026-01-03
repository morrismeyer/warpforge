# SnakeGrinder bootstrap script.
# No external dependencies. No pip.

import sys
import platform


def self_test():
    # Keep the output simple and stable for CLI validation.
    return {
        "status": "ok",
        "python_version": sys.version,
        "platform": platform.platform(),
    }
