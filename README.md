# Glider Dashboard

[![GitHub release](https://img.shields.io/github/v/release/voto-ocean-knowledge/glider_dashboard)](https://github.com/voto-ocean-knowledge/glider_dashboard/releases)
[![PyPI](https://img.shields.io/pypi/v/glider-dashboard)](https://pypi.org/project/glider-dashboard/)
[![Tests](https://github.com/voto-ocean-knowledge/glider_dashboard/actions/workflows/python-package.yml/badge.svg)](https://github.com/voto-ocean-knowledge/glider_dashboard/actions/workflows/python-package.yml)

A Python program for visualising and interacting with large amounts of ocean glider data,
downloaded from the [Voice of the Ocean ERDDAP](https://erddap.observations.voiceoftheocean.org/erddap/)
(and optionally the IOOS glider GDAC).

## Quick start

```bash
pip install glider-dashboard
glider-dashboard serve --show
```

This downloads a few recent missions (by default the three latest in the Bornholm Basin)
into `./glider_dashboard_data` and opens the dashboard at <http://localhost:5006>.

## Configuration

Everything that controls what is downloaded and where it is stored lives in one YAML file:

```bash
glider-dashboard init-config      # writes an annotated glider_dashboard.yaml
```

Edit it — for example to pick specific missions:

```yaml
data_dir: glider_dashboard_data
selection:
  dataset_ids: [SEA078_M43, SEA067_M58]
```

A `glider_dashboard.yaml` in the working directory is used automatically; otherwise pass
`--config path/to/file.yaml` or set `GLIDER_DASHBOARD_CONFIG`. Keys you leave out keep
their defaults.

## Commands

| Command | What it does |
| --- | --- |
| `glider-dashboard init-config [PATH]` | Write the default config file |
| `glider-dashboard download` | Download and convert the selected missions |
| `glider-dashboard serve` | Download, then serve the dashboard (`--no-download` to skip) |

`serve` runs `panel serve` on the app, so every browser session gets its own state.
Arguments after `--` are passed on to `panel serve`, e.g.
`glider-dashboard serve -- --autoreload`. For a production server that refreshes its data
every two hours: `glider-dashboard serve --restart-hours 2 --config production.yaml`.

## Development

```bash
git clone https://github.com/voto-ocean-knowledge/glider_dashboard.git
cd glider_dashboard
pip install -e ".[test]"
pytest
```

The tests use small synthetic datasets and need no network access. Importing the package
never downloads data, so the API documentation builds without it.

[Code](https://github.com/voto-ocean-knowledge/glider_dashboard) \|
[Issues](https://github.com/voto-ocean-knowledge/glider_dashboard/issues) \|
[Documentation](https://glider-dashboard.readthedocs.io/en/docs/)
