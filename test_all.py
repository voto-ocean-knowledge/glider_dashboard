import glider_dashboard as gdb
import numpy as np
import time

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
    gdb.utils.year = 2023
    metadata, all_datasets = gdb.utils.filter_metadata()
    print(len(gdb.metadata))
    print(len(metadata))

def test_with_event():
    glider_dashboard = gdb.GliderDashboard()
    glider_dashboard.markdown.object = """\
    # Baltic Inflows
    Baltic Inflows are transporting salt and oxygen into the depth of the Baltic Sea.
    """
    # for i in range(10,20):
    glider_dashboard.startX = np.datetime64("2024-01-15")
    glider_dashboard.endX = np.datetime64(f"2024-01-18")
    glider_dashboard.pick_startX = np.datetime64("2024-01-15")
    glider_dashboard.pick_endX = np.datetime64(f"2024-01-18")

    time.sleep(5)
    print("event:plot reloaded")
    glider_dashboard.startX = np.datetime64("2024-01-15")
    glider_dashboard.endX = np.datetime64("2024-03-20")
    # glider_dashboard.annotations.append(text_annotation)
    glider_dashboard.pick_variable = "oxygen_concentration"