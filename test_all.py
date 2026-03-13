```python
import functools
import time
import timeit
from os.path import join

import numpy as np
import panel as pn
import glider_explorer as gdb

# Define constants
OUTPATH = "./test_plots"

def test_import():
    """
    Test if the correct Python environment is activated and packages are installed.
    """
    assert 1 == 1

def test_dataset_is_loaded():
    """
    Test if the dataset is loaded.
    """
    assert len(gdb.metadata) > 0
    print(gdb.metadata)

def test_filter_metadata():
    """
    Test filtering metadata.
    """
    gdb.utils.year = 2024
    metadata = gdb.utils.filter_metadata()
    print(len(gdb.metadata))
    print(len(metadata))

def test_glider_dashboard():
    """
    Test the GliderDashboard class.
    """
    gdb.GliderDashboard()

def test_create_dynmap():
    """
    Test creating a dynamic map.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_variable = "salinity"
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "salinity.png"))

def test_create_dynmap_temperature():
    """
    Test creating a dynamic map for temperature.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-18")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_variable = "temperature"
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "temperature.png"))

def test_load_viewport_datasets():
    """
    Test loading viewport datasets.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    t = timeit.Timer(
        functools.partial(GDB.load_viewport_datasets, (GDB.pick_startX, GDB.pick_endX))
    )
    print("load_viewport_datasets takes:", t.timeit(10) / 10)

def test_create_dynmap_mld():
    """
    Test creating a dynamic map for MLD.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_mld = True
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "mld.png"))

def test_create_dynmap_scatter_plot():
    """
    Test creating a dynamic map for scatter plot.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_TS = True
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "TS.png"))

def test_create_dynmap_profile_plots():
    """
    Test creating a dynamic map for profile plots.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_profiles = True
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "profiles.png"))

def test_create_dynmap_cnorm():
    """
    Test creating a dynamic map with a custom colormap.
    """
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.cnorm = "eq_hist"
    dyn = GDB.create_dynmap()
    dyn.save(join(OUTPATH, "eq_hist.png"))

if __name__ == "__main__":
    test_import()
    test_dataset_is_loaded()
    test_filter_metadata()
    test_glider_dashboard()
    test_create_dynmap()
    test_create_dynmap_temperature()
    test_load_viewport_datasets()
    test_create_dynmap_mld()
    test_create_dynmap_scatter_plot()
    test_create_dynmap_profile_plots()
    test_create_dynmap_cnorm()
```