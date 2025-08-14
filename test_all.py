import glider_explorer as gdb
import numpy as np
import time
import panel as pn
import pytest
import timeit
import functools
from os.path import join

# tests to add:
# 1. Are datapoints drawn?
# 2. Are datapoints still drawn if zoomed all the way in to a single day?
# 3. Can the x_range be defined by (url-) parameters?
# 4. Tipp: I could probably implement tests the same way I implement events.
outpath = './test_plots'

def test_import():
    # tests if correct python environment is activated, syntax is reasonable, packages installed...
    assert 1==1

def test_dataset_is_loaded():
    # data is loaded
    assert len(gdb.metadata)>0
    print(gdb.metadata)

def test_filter_metadata():
    gdb.utils.year = 2024
    metadata, all_datasets = gdb.utils.filter_metadata()
    print(len(gdb.metadata))
    print(len(metadata))


def test_salinity():
    GDB = gdb.GliderDashboard()
    #GDB.startX = np.datetime64('2024-03-01')
    #GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64('2024-01-01')
    GDB.pick_endX = np.datetime64('2024-12-19')
    GDB.pick_variable = 'salinity'
    #import pdb; pdb.set_trace();

    # create output for variable salinity
    t1 = time.perf_counter()
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    #myapp = pn.panel(dyn)
    pn.pane.HoloViews(dyn).save(join(outpath, 'salinity.png'))
    t2 = time.perf_counter()
    print('creating the first serve took',t2-t1)

def test_temperature():
    GDB = gdb.GliderDashboard()
    #GDB.startX = np.datetime64('2024-03-01')
    #GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64('2024-01-18')
    GDB.pick_endX = np.datetime64('2024-12-19')

    # create output for variable temperature
    t1 = time.perf_counter()
    GDB.pick_variable = 'temperature'
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    pn.pane.HoloViews(dyn).save(join(outpath, 'temperature.png'))
    t2 = time.perf_counter()
    print('creating the second serve took',t2-t1)
    t = timeit.Timer(functools.partial(GDB.load_viewport_datasets, (GDB.pick_startX, GDB.pick_endX)))
    print('load_viewport_datasets takes:', t.timeit(10)/10)

    # activate mld
    t1 = time.perf_counter()
    GDB.pick_mld = True
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    #myapp = pn.panel(dyn)
    pn.pane.HoloViews(dyn).save(join(outpath, 'mld.png'))
    t2 = time.perf_counter()
    print('creating the third serve took',t2-t1)
    GDB.pick_mld = False

    # activate scatter plot
    GDB.pick_TS = True
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    myapp = pn.panel(dyn)
    pn.pane.HoloViews(dyn).save(join(outpath, 'TS.png'))
    GDB.pick_TS = False

    # activate profile plots
    GDB.pick_profiles = True
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    myapp = pn.panel(dyn)
    pn.pane.HoloViews(dyn).save(join(outpath, 'profiles.png'))
    GDB.pick_profiles = False

    # toggle to DatasetID
    # GDB.pick_toggle = 'DatasetID'
    # dyn = GDB.create_dynmap().opts(width=500, height=500)
    # myapp = pn.panel(dyn)
    # pn.pane.HoloViews(dyn).save(join(outpath, 'DatasetID.png'))
    # GDB.pick_toggle = 'SAMBA obs.'

    # colorbar
    GDB.cnorm = 'eq_hist'
    dyn = GDB.create_dynmap().opts(width=500, height=500)
    myapp = pn.panel(dyn)
    pn.pane.HoloViews(dyn).save(join(outpath, 'eq_hist.png'))
    GDB.pick_cnorm = 'linear'

    assert 1==1
