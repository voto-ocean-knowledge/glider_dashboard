import dataclasses
from pathlib import Path

import pytest
import yaml

from glider_dashboard.config import CONFIG_ENV_VAR, Config, default_config_text, load_config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)


def test_defaults_without_config_file(tmp_path):
    config = load_config()
    assert config.source is None
    assert config.data_dir == tmp_path / "glider_dashboard_data"
    assert config.gdac.enabled is False


def test_packaged_yaml_matches_dataclass_defaults(tmp_path):
    """default_config.yaml and the dataclass defaults must not drift apart."""
    from_yaml = Config.from_dict(yaml.safe_load(default_config_text()), base_dir=tmp_path)
    builtin = Config.from_dict({}, base_dir=tmp_path)
    assert dataclasses.asdict(from_yaml) == dataclasses.asdict(builtin)


def test_partial_config_overrides_defaults(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    path = config_dir / "my.yaml"
    path.write_text("data_dir: data\nselection:\n  latest: 1\n  basins: [Åland Sea]\n")

    config = load_config(path)
    assert config.source == path
    assert config.data_dir == config_dir / "data"  # relative to the config file
    assert config.selection.latest == 1
    assert config.selection.basins == ["Åland Sea"]
    assert config.selection.project == "SAMBA"  # untouched default


def test_config_found_via_env_and_cwd(tmp_path, monkeypatch):
    (tmp_path / "glider_dashboard.yaml").write_text("selection: {latest: 2}\n")
    assert load_config().selection.latest == 2

    other = tmp_path / "other.yaml"
    other.write_text("selection: {latest: 5}\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(other))
    assert load_config().selection.latest == 5


@pytest.mark.parametrize(
    "text", ["data_dri: x\n", "selection: {basin: [Bornholm Basin]}\n"]
)
def test_unknown_keys_are_rejected(tmp_path, text):
    path = Path("bad.yaml")
    path.write_text(text)
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
