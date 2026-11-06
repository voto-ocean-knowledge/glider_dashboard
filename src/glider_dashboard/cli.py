"""Command line interface: ``glider-dashboard <command>``.

Commands
--------
``init-config``
    Write the annotated default configuration to a file to edit.
``download``
    Download and convert the data selected in the configuration.
``serve``
    Download (unless ``--no-download``) and start the dashboard server.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

from . import __version__
from .config import CONFIG_ENV_VAR, DEFAULT_CONFIG_NAME, Config, default_config_text, load_config

logger = logging.getLogger("glider_dashboard")


def init_config(args) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        logger.error("%s already exists, use --force to overwrite", path)
        return 1
    path.write_text(default_config_text())
    print(f"Wrote {path}")
    return 0


def download(args) -> int:
    from .download import update_data  # heavy imports only when needed

    config = load_config(args.config)
    _log_config(config)
    failed = update_data(config)
    if failed:
        logger.error("Failed datasets: %s", ", ".join(failed))
        return 1
    return 0


def panel_command(config: Config, args) -> list[str]:
    """``panel serve`` command line for the app."""
    app = resources.files("glider_dashboard").joinpath("app.py")
    cmd = [sys.executable, "-m", "panel", "serve", str(app)]
    cmd += ["--port", str(args.port or config.server.port)]
    for origin in config.server.allow_websocket_origin:
        cmd += ["--allow-websocket-origin", origin]
    cmd += config.server.extra_args
    if args.show:
        cmd.append("--show")
    return cmd + args.panel_args


def serve(args) -> int:
    config = load_config(args.config)
    _log_config(config)
    env = dict(os.environ)
    if config.source is not None:
        env[CONFIG_ENV_VAR] = str(config.source)

    while True:
        if not args.no_download:
            from .download import update_data

            update_data(config)
        cmd = panel_command(config, args)
        logger.info("Running %s", " ".join(cmd))
        process = subprocess.Popen(cmd, env=env)
        try:
            if args.restart_hours is None:
                return process.wait()
            try:
                process.wait(timeout=args.restart_hours * 3600)
                return process.returncode
            except subprocess.TimeoutExpired:
                logger.info("Restarting the server to refresh the data")
                process.terminate()
                process.wait()
                time.sleep(1)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return 0


def _log_config(config: Config) -> None:
    logger.info("Config: %s", config.source or "built-in defaults")
    logger.info("Data directory: %s", config.data_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glider-dashboard",
        description="Interactive dashboard for ocean glider data from ERDDAP.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_config_option(p):
        p.add_argument(
            "-c",
            "--config",
            help=f"YAML config file (default: ${CONFIG_ENV_VAR} or ./{DEFAULT_CONFIG_NAME}"
            " if present, else built-in defaults)",
        )

    p = commands.add_parser("init-config", help="write the default config file")
    p.add_argument("path", nargs="?", default=DEFAULT_CONFIG_NAME)
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=init_config)

    p = commands.add_parser("download", help="download and convert the selected data")
    add_config_option(p)
    p.set_defaults(func=download)

    p = commands.add_parser(
        "serve",
        help="start the dashboard server",
        description="Start the dashboard server. Arguments after `--` are passed to `panel serve`.",
    )
    add_config_option(p)
    p.add_argument("--port", type=int, help="override server.port from the config")
    p.add_argument("--no-download", action="store_true", help="only use data already on disk")
    p.add_argument("--show", action="store_true", help="open the dashboard in a browser")
    p.add_argument(
        "--restart-hours",
        type=float,
        help="re-download data and restart the server at this interval",
    )
    p.add_argument("panel_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.set_defaults(func=serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "panel_args", None) and args.panel_args[0] == "--":
        args.panel_args = args.panel_args[1:]
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)
