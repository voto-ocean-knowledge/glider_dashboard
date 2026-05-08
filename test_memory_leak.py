import os
from os.path import join

import numpy as np
import pandas as pd
import psutil

import glider_explorer as gdb

# outpath = "./test_plots"


def test_salinity(tmp_path, iteration_number="not applicable"):
    GDB = gdb.GliderDashboard()
    # GDB.startX = np.datetime64('2024-03-01')
    # GDB.endX = np.datetime64('2024-05-01')

    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_variable = "salinity"
    # import pdb; pdb.set_trace();

    # create output for variable salinity
    dyn = GDB.create_dynmap()  # .opts(width=500, height=500)
    # myapp = pn.panel(dyn)
    dyn.save(join(tmp_path, "salinity.png"))

    process = psutil.Process(os.getpid())
    ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"iteration number: {iteration_number}, ram used: {round(ram_used, 2)}")
    if iteration_number in [100, 119]:
        from pympler import muppy, summary

        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

        from pympler import tracker

        mem = tracker.SummaryTracker()
        memory = pd.DataFrame(
            mem.create_summary(), columns=["object", "number_of_objects", "memory"]
        )
        memory["mem_per_object"] = memory["memory"] / memory["number_of_objects"]
        print(memory.sort_values("memory", ascending=False).head(10))
        print(memory.sort_values("mem_per_object", ascending=False).head(10))

        dataframes = {
            name: obj
            for name, obj in globals().items()
            if isinstance(obj, pd.DataFrame)
        }
        print(dataframes)


def test_memory_leak(tmp_path):
    for i in range(0, 20):
        test_salinity(tmp_path, iteration_number=i)
    print(f"salinity function ran {i} times, please check for memory increase")
