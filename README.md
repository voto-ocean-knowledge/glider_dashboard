# Glider_explorer

[![GitHub release](https://img.shields.io/github/v/release/Klankers/glider_dashboard)](https://github.com/Klankers/glider_dashboard/releases)
[![Tests](https://github.com/Klankers/glider_dashboard/actions/workflows/python-package.yml/badge.svg)](https://github.com/Klankers/glider_dashboard/actions/workflows/python-package.yml)

A python program for visualizing and interacting with large amounts of ocean glider data.

## Usage notes

To serve this in a multi-user environment, use `panel serve glider_explorer.py`.
If this is served via the `pn.serve()` function from within the script, the session state will be shared accross users instead (which would be irritating).

```bash
python initialize.py
python parquet_converter.py
panel serve glider_explorer.py --port 5006 --warm --use-xheaders --allow-websocked-origin='*' #--admin --profiler=snakeviz
```

A bash file is included for simple setup and execution. For a local execution, try:

```bash
bash start_glider_dashboard.bash
```

[Code](https://github.com/voto-ocean-knowledge/glider_dashboard) \|
[Issues](https://github.com/voto-ocean-knowledge/glider_dashboard/issues) \|
[Documentation](https://glider-dashboard.readthedocs.io/en/docs/)
