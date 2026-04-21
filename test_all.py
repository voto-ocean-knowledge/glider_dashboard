import functools
import time
import timeit
from os.path import join

import numpy as np
import panel as pn
import polars as pl

import glider_explorer as gdb

# tests to add:
# 1. Are datapoints drawn?
# 2. Are datapoints still drawn if zoomed all the way in to a single day?
# 3. Can the x_range be defined by (url-) parameters?
# 4. Tipp: I could probably implement tests the same way I implement events.
# outpath = "./test_plots"


def test_import():
    # tests if correct python environment is activated, syntax is reasonable, packages installed...
    assert 1 == 1


def test_dataset_is_loaded():
    # data is loaded
    assert len(gdb.lod.metadata) > 0
    print(gdb.lod.metadata)


def test_filter_metadata():
    gdb.utils.year = 2024
    metadata = gdb.utils.filter_metadata()
    print(len(gdb.lod.metadata))
    print(len(metadata))


def test_salinity(tmp_path):
    # breakpoint()
    GDB = gdb.GliderDashboard()
    # GDB.startX = np.datetime64('2024-03-01')
    # GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_variable = "salinity"
    # import pdb; pdb.set_trace();

    # create output for variable salinity
    t1 = time.perf_counter()
    dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    # myapp = pn.panel(dyn)
    dyn.save(join(tmp_path, "salinity.png"))
    t2 = time.perf_counter()
    print("creating the first serve took", t2 - t1)


def test_temperature(tmp_path):
    GDB = gdb.GliderDashboard()
    # GDB.startX = np.datetime64('2024-03-01')
    # GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64("2024-01-18")
    GDB.pick_endX = np.datetime64("2024-12-19")

    # create output for variable temperature
    t1 = time.perf_counter()
    GDB.pick_variable = "temperature"
    dyn = GDB.create_dynmap()
    dyn.save(join(tmp_path, "temperature.png"))
    t2 = time.perf_counter()
    print("creating the second serve took", t2 - t1)
    t = timeit.Timer(
        functools.partial(GDB.load_viewport_datasets, (GDB.pick_startX, GDB.pick_endX))
    )
    print("load_viewport_datasets takes:", t.timeit(10) / 10)

    # activate mld
    t1 = time.perf_counter()
    # GDB.pick_mld = True
    # dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    # # myapp = pn.panel(dyn)
    # dyn.save(join(outpath, "mld.png"))
    # t2 = time.perf_counter()
    # print("creating the third serve took", t2 - t1)
    # GDB.pick_mld = False

    # activate scatter plot
    GDB.pick_scatter_bool = True
    GDB.pick_scatter_x = "temperature"
    GDB.pick_scatter_y = "pressure"
    dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    myapp = pn.panel(dyn)
    dyn.save(join(tmp_path, "TS.png"))
    GDB.pick_TS = False

    # activate profile plots
    # GDB.pick_scatter_bool = True
    # dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    # myapp = pn.panel(dyn)
    # dyn.save(join(outpath, "profiles.png"))
    # GDB.pick_profiles = False

    # toggle to DatasetID
    # GDB.pick_toggle = 'DatasetID'
    # dyn = GDB.create_dynmap().opts(width=500, height=500)
    # myapp = pn.panel(dyn)
    # pn.pane.HoloViews(dyn).save(join(outpath, 'DatasetID.png'))
    # GDB.pick_toggle = 'SAMBA obs.'

    # colorbar
    GDB.cnorm = "eq_hist"
    dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    myapp = pn.panel(dyn)
    dyn.save(join(tmp_path, "eq_hist.png"))
    GDB.pick_cnorm = "linear"

    assert 1 == 1

def test_update_markdown():
    import utils
    # breakpoint()
    GDB = gdb.GliderDashboard()
    x_range = (np.datetime64("2024-01-18"), np.datetime64("2024-12-19"))
    y_range = (np.float64(0), np.float64(50.5))
    assert "pick_toggle" in dir(GDB)
    assert GDB.pick_toggle == "SAMBA obs."  #   Should default to SAMBA
    assert GDB.pick_variables == ["temperature"]

    #   To make data_on_view, we need to concat the LazyFrames in varlist (get_xsection_raster)
    GDB.get_xsection_raster(x_range = x_range, y_range = y_range, x = None, y = None)
    assert type(GDB.data_in_view) is pl.LazyFrame

    output = GDB.update_markdown(x_range = x_range, y_range = y_range)

    # Pulling from data on the server, these values should not change.
    assert "Bornholm Basin from 2024-01-18 00:00:00 to 2024-12-19 00:00:00" in output
    assert "Number of Profiles    | 3895" in output
    assert "| temperature | 2.19 / 10.85 / 5.32 / 2.21 |" in output
    assert "<tr><td>AD2CP_make_model</td><td>Nortek AD2CP</td></tr>" in output
