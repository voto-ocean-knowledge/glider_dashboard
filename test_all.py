import glider_dashboard as gdb
import numpy as np
import time
import panel as pn
import pytest
import timeit
import functools

# tests to add:
# 1. Are datapoints drawn?
# 2. Are datapoints still drawn if zoomed all the way in to a single day?
# 3. Can the x_range be defined by (url-) parameters?
# 4. Tipp: I could probably implement tests the same way I implement events.

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


def test_with_event():
    GDB = gdb.GliderDashboard()
    #GDB.startX = np.datetime64('2024-03-01')
    #GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64('2024-04-18')
    GDB.pick_endX = np.datetime64('2024-04-19')
    GDB.pick_variable = 'salinity'
    #import pdb; pdb.set_trace();

    t1 = time.perf_counter()
    dyn = GDB.create_dynmap()
    myapp = pn.panel(dyn)
    myapp.save('panel_output1.png')
    t2 = time.perf_counter()
    print('creating the first serve took',t2-t1)

    t1 = time.perf_counter()
    GDB.pick_variable = 'temperature'
    dyn = GDB.create_dynmap()
    myapp = pn.panel(dyn)
    myapp.save('panel_output2.png')
    t2 = time.perf_counter()
    print('creating the second serve took',t2-t1)
    t = timeit.Timer(functools.partial(GDB.load_viewport_datasets, (GDB.pick_startX, GDB.pick_endX)))
    print('load_viewport_datasets takes:', t.timeit(10)/10)
    assert 1==1
