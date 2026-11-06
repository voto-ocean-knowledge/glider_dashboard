"""Configuration for the Glider Dashboard.

All settings that used to be hard-coded (ERDDAP servers, the local data
directory, which missions to download) live in a single YAML file. The defaults
are shipped with the package in ``default_config.yaml``; a user config only
needs to contain the keys it changes.

The config file is found, in order of preference, from

1. an explicit path (``--config`` on the command line),
2. the ``GLIDER_DASHBOARD_CONFIG`` environment variable,
3. ``glider_dashboard.yaml`` in the current working directory,

and otherwise the packaged defaults are used.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

#: Environment variable used to hand the config path to ``panel serve``.
CONFIG_ENV_VAR = "GLIDER_DASHBOARD_CONFIG"

#: Config file picked up automatically from the working directory.
DEFAULT_CONFIG_NAME = "glider_dashboard.yaml"


@dataclass
class VotoConfig:
    """Voice of the Ocean ERDDAP server."""

    server: str = "https://erddap.observations.voiceoftheocean.org/erddap"
    adcp_files_url: str = (
        "https://erddap.observations.voiceoftheocean.org/erddap/files/gliderad2cp_files"
    )
    chunk_days: float | None = 7


@dataclass
class SelectionConfig:
    """Which VOTO missions are downloaded and shown."""

    dataset_ids: list[str] = field(default_factory=list)
    project: str | None = "SAMBA"
    basins: list[str] = field(default_factory=lambda: ["Bornholm Basin"])
    years: list[int] = field(default_factory=list)
    latest: int | None = 3
    exclude_dataset_ids: list[str] = field(
        default_factory=lambda: ["nrt_SEA067_M15", "nrt_SEA079_M14", "nrt_SEA061_M63"]
    )


@dataclass
class GdacConfig:
    """IOOS glider GDAC server."""

    enabled: bool = False
    server: str = "https://gliders.ioos.us/erddap"
    year: int | None = 2026
    month: int | None = 4
    max_datasets: int = 90
    max_variables: int = 200
    exclude_institutions: list[str] = field(
        default_factory=lambda: [
            "C-PROOF",
            "OOI Coastal & Global Scale Nodes (CGSN)",
        ]
    )


@dataclass
class ServerConfig:
    """Options passed on to ``panel serve``."""

    port: int = 5006
    allow_websocket_origin: list[str] = field(default_factory=lambda: ["*"])
    extra_args: list[str] = field(default_factory=lambda: ["--warm", "--use-xheaders"])


@dataclass
class Config:
    """Complete dashboard configuration."""

    data_dir: Path = Path("glider_dashboard_data")
    voto: VotoConfig = field(default_factory=VotoConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    gdac: GdacConfig = field(default_factory=GdacConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    #: File the config was read from, ``None`` for the built-in defaults.
    source: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | None = None) -> Config:
        """Build a config from a (possibly partial) nested dictionary.

        Unknown keys raise a ``ValueError`` so that typos do not silently fall
        back to defaults. A relative ``data_dir`` is resolved against
        ``base_dir`` (defaults to the current working directory).
        """
        data = dict(data or {})
        sections = {
            "voto": VotoConfig,
            "selection": SelectionConfig,
            "gdac": GdacConfig,
            "server": ServerConfig,
        }
        kwargs = {}
        for name, section_cls in sections.items():
            kwargs[name] = _build_section(section_cls, data.pop(name, None) or {}, name)

        data_dir = Path(data.pop("data_dir", cls.data_dir)).expanduser()
        if not data_dir.is_absolute():
            data_dir = (base_dir or Path.cwd()) / data_dir
        kwargs["data_dir"] = data_dir

        if data:
            raise ValueError(f"Unknown config keys: {sorted(data)}")
        return cls(**kwargs)


def _build_section(section_cls, values: dict, section_name: str):
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section_name}': {sorted(unknown)}")
    return section_cls(**values)


def default_config_text() -> str:
    """Return the annotated default config shipped with the package."""
    return resources.files("glider_dashboard").joinpath("default_config.yaml").read_text()


def find_config_file(path: str | os.PathLike | None = None) -> Path | None:
    """Locate the config file to use, see the module docstring for the order."""
    if path is not None:
        return Path(path)
    if os.environ.get(CONFIG_ENV_VAR):
        return Path(os.environ[CONFIG_ENV_VAR])
    if Path(DEFAULT_CONFIG_NAME).is_file():
        return Path(DEFAULT_CONFIG_NAME)
    return None


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the dashboard configuration.

    Parameters
    ----------
    path : str or path-like, optional
        YAML file to read. If omitted, :func:`find_config_file` decides.

    Returns
    -------
    Config
    """
    config_file = find_config_file(path)
    if config_file is None:
        return Config.from_dict({})
    config_file = config_file.expanduser().resolve()
    with open(config_file) as f:
        data = yaml.safe_load(f) or {}
    config = Config.from_dict(data, base_dir=config_file.parent)
    config.source = config_file
    return config
