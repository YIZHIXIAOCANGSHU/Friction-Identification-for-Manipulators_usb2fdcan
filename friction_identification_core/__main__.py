from __future__ import annotations

import argparse

from friction_identification_core.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.results import log_info
from friction_identification_core.workflow import run_step_torque


run_step_torque_scan = run_step_torque


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential step-torque sweep CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("step", "default"),
        default="step",
        help="Primary runtime mode. `default` aliases to `step`.",
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


def _normalize_mode(mode: str) -> str:
    if mode == "default":
        return "step"
    return str(mode)


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        config = load_config(args.config)
        config = apply_overrides(config, output=args.output, motors=args.motors)
        mode = _normalize_mode(str(args.mode))
        if mode != "step":
            raise ValueError(f"Unsupported mode: {mode}")
        run_step_torque_scan(config, show_rerun_viewer=True)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    except KeyboardInterrupt:
        log_info("Interrupted by user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
