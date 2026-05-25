from __future__ import annotations

from mit_sender.app import main
from mit_sender.commands import SelectedMotorCommand, build_uniform_commands
from mit_sender.ui import MitSenderWindow


if __name__ == "__main__":
    raise SystemExit(main())
