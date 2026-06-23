# Run the actual test to get a baseline memory reading
import gc
import os
import sys
from os.path import join

import numpy as np
import psutil

sys.path.insert(0, "/home/martin/MHW/glider_dashboard")
os.chdir("/home/martin/MHW/glider_dashboard")

from holoviews.streams import Stream

import glider_explorer as gdb

process = psutil.Process(os.getpid())

for i in range(8):
    GDB = gdb.GliderDashboard()
    GDB.pick_startX = np.datetime64("2024-01-01")
    GDB.pick_endX = np.datetime64("2024-12-19")
    GDB.pick_variable = "salinity"
    dyn = GDB.create_dynmap()
    dyn.save(join("./test_plots/", "salinity.png"))

    # Explicitly delete to let GC work
    del dyn
    del GDB
    gc.collect()

    ram_used = process.memory_info().rss / (1024 * 1024)
    registry_size = len(Stream.registry)
    globals

import sys


def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


for name, size in sorted(
    ((name, sys.getsizeof(value)) for name, value in list(locals().items())),
    key=lambda x: -x[1],
)[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


import pdb

pdb.set_trace()
