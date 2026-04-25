from __future__ import annotations

import argparse

from friction_identification_core.runtime_config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.results import log_info
from friction_identification_core.workflow import (
    run_breakaway,
    run_compensation,
    run_identify_all,
    run_inertia,
    run_speed_hold,
)

MODE_CHOICES = ("identify-all", "compensation", "breakaway", "speed-hold", "inertia")


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIT identify-all friction and inertia identification CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="identify-all",
        help="Runtime mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--motors",
        default=None,
        help="Target motor ids, for example '1,3,4' or 'all'.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        config = load_config(args.config)
        config = apply_overrides(config, output=args.output, motors=args.motors)
        mode = str(args.mode)
        if mode == "identify-all":
            run_identify_all(config, show_rerun_viewer=True)
        elif mode == "compensation":
            run_compensation(config, show_rerun_viewer=True)
        elif mode == "breakaway":
            run_breakaway(config, show_rerun_viewer=True)
        elif mode == "speed-hold":
            run_speed_hold(config, show_rerun_viewer=True)
        elif mode == "inertia":
            run_inertia(config, show_rerun_viewer=True)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported mode: {mode}")
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    except KeyboardInterrupt:
        log_info("Interrupted by user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
