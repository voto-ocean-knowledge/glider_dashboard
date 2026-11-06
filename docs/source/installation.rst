Installation
============

Glider Dashboard requires Python 3.11 or newer. Install it from PyPI:

.. code-block:: bash

   pip install glider-dashboard

or, for development, from a clone of the repository:

.. code-block:: bash

   git clone https://github.com/voto-ocean-knowledge/glider_dashboard.git
   cd glider_dashboard
   pip install -e ".[test]"

Conda users can create an environment with all dependencies first:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate glider_dashboard
   pip install --no-deps -e .

Running the dashboard
---------------------

.. code-block:: bash

   glider-dashboard serve --show

This downloads the missions selected in the configuration (by default the
three most recent missions in the Bornholm Basin) into
``./glider_dashboard_data`` and opens the dashboard at http://localhost:5006.

The steps can also be run separately:

.. code-block:: bash

   glider-dashboard download          # fetch and convert data only
   glider-dashboard serve --no-download

Arguments after ``--`` are passed on to ``panel serve``, e.g.
``glider-dashboard serve -- --autoreload`` during development. For a
long-running server that refreshes its data every two hours, use
``glider-dashboard serve --restart-hours 2``.

Configuration
-------------

All settings — ERDDAP servers, the data directory, which missions to download
and the server options — are read from a YAML file. Write the annotated
defaults to ``glider_dashboard.yaml`` with

.. code-block:: bash

   glider-dashboard init-config

edit it, and it is picked up automatically from the working directory (or pass
``--config path/to/file.yaml``, or set ``GLIDER_DASHBOARD_CONFIG``). Keys left
out of the file keep their default values. The defaults are:

.. literalinclude:: ../../src/glider_dashboard/default_config.yaml
   :language: yaml

For example, to show two specific missions only:

.. code-block:: yaml

   selection:
     dataset_ids: [SEA078_M43, SEA067_M58]

or to show all SAMBA missions since 2024 in two observatories (a large
download):

.. code-block:: yaml

   data_dir: /data/voto_erddap_data_cache
   selection:
     basins: [Bornholm Basin, Eastern Gotland]
     years: [2024, 2025, 2026]
     latest: null
