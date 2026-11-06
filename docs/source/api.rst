API Reference
=============

Importing any of these modules does not download or read data. Data is only
loaded when a :class:`~glider_dashboard.catalog.DataCatalog` is built, which the
served app (:mod:`glider_dashboard.app`) and the command line interface do.

Dashboard
---------

.. automodule:: glider_dashboard.dashboard
   :members:
   :show-inheritance:

Data catalog
------------

.. automodule:: glider_dashboard.catalog
   :members:

Datasets and mission selection
------------------------------

.. automodule:: glider_dashboard.datasets
   :members:

Downloading
-----------

.. automodule:: glider_dashboard.download
   :members:

Processing
----------

.. automodule:: glider_dashboard.processing
   :members:

Configuration
-------------

.. automodule:: glider_dashboard.config
   :members:

Command line interface
----------------------

.. automodule:: glider_dashboard.cli
   :members: main
