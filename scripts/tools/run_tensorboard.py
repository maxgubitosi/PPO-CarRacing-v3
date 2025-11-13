#!/usr/bin/env python3
"""Wrapper around TensorBoard CLI that suppresses pkg_resources deprecation warning."""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

from tensorboard import main as tensorboard_main  # noqa: E402


def main() -> None:
    tensorboard_main.run_main()


if __name__ == "__main__":
    # Ensure argv looks like tensorboard saw it directly.
    sys.argv = ["tensorboard", *sys.argv[1:]]
    main()
