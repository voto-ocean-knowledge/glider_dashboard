import sys

from glider_dashboard.cli import build_parser, main, panel_command
from glider_dashboard.config import Config


def test_init_config_writes_default(tmp_path, capsys):
    target = tmp_path / "conf.yaml"
    assert main(["init-config", str(target)]) == 0
    assert "data_dir:" in target.read_text()
    assert main(["init-config", str(target)]) == 1  # refuses to overwrite


def test_panel_command():
    args = build_parser().parse_args(["serve", "--port", "7000", "--show", "--", "--dev"])
    args.panel_args = args.panel_args[1:]
    cmd = panel_command(Config(), args)
    assert cmd[:4] == [sys.executable, "-m", "panel", "serve"]
    assert cmd[4].endswith("app.py")
    assert cmd[cmd.index("--port") + 1] == "7000"
    assert cmd[-2:] == ["--show", "--dev"]
