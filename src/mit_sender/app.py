from __future__ import annotations

from mit_sender.ui import MitSenderWindow, create_app


def main() -> int:
    app = create_app()
    window = MitSenderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
