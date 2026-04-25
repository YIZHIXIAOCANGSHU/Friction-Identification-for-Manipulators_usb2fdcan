from pathlib import Path
import sys

try:
    from send.damiao import *  # noqa: F401,F403
    from send.damiao import build_socketcan_parser as build_parser
    from send.damiao import socketcan_main as main
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from send.damiao import *  # noqa: F401,F403
    from send.damiao import build_socketcan_parser as build_parser
    from send.damiao import socketcan_main as main


if __name__ == "__main__":
    raise SystemExit(main())
