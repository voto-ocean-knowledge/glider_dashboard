"""Smoke tests of the catalog and dashboard on synthetic data (no network)."""

import datashader as dsh
import holoviews as hv
from holoviews.operation.datashader import rasterize
import numpy as np
import panel as pn
import polars as pl

from glider_dashboard.dashboard import GliderDashboard, cbar_range_name, create_metadata_table

hv.extension("bokeh")


def test_catalog_uses_delayed_data_when_available(catalog):
    assert catalog.mission_ids == ["nrt_SEA001_M1", "nrt_SEA002_M2", "nrt_SEA003_M3"]
    assert catalog.data_id("nrt_SEA001_M1") == "delayed_SEA001_M1"
    assert catalog.data_id("nrt_SEA002_M2") == "nrt_SEA002_M2"
    assert {"temperature", "salinity", "time", "depth"} <= set(catalog.variables)
    assert isinstance(catalog.frame("nrt_SEA001_M1", small=True), pl.LazyFrame)


def test_catalog_skips_missions_without_files(catalog, data_dir, metadata, voto_datasets):
    (data_dir / "nrt_SEA002_M2_small.parquet").unlink()
    reduced = type(catalog)(data_dir, metadata, metadata, voto_datasets)
    assert "nrt_SEA002_M2" not in reduced.mission_ids
    assert "nrt_SEA002_M2" not in reduced.metadata.index


def test_dashboard_choices_come_from_catalog(catalog):
    dashboard = GliderDashboard(catalog)
    assert dashboard.param.pick_variables.objects == catalog.variables
    assert dashboard.param.pick_dsids.objects == {
        "SEA001_M1": "nrt_SEA001_M1",
        "SEA002_M2": "nrt_SEA002_M2",
        "SEA003_M3": "nrt_SEA003_M3",
    }
    for variable in catalog.variables:
        assert cbar_range_name(variable) in dashboard.param


def test_colorbar_slider_triggers_redraw(catalog):
    dashboard = GliderDashboard(catalog)
    events = []
    dashboard.param.watch(lambda e: events.append(e.name), "refresh_plots")
    setattr(dashboard, cbar_range_name("temperature"), (2, 8))
    assert events == ["refresh_plots"]


def test_viewport_selects_observatory_missions(catalog):
    dashboard = GliderDashboard(catalog, pick_basin="Bornholm Basin")
    x_range = (np.datetime64("2024-01-10"), np.datetime64("2024-02-01"))
    assert list(dashboard.load_viewport_datasets(x_range).index) == ["nrt_SEA001_M1"]
    assert len(dashboard.load_viewport_datasets((None, None))) == 2

    dashboard.pick_toggle = "DatasetID"
    dashboard.pick_dsids = ["nrt_SEA003_M3"]
    assert list(dashboard.load_viewport_datasets((None, None)).index) == ["nrt_SEA003_M3"]


def test_section_data_and_statistics(catalog):
    dashboard = GliderDashboard(catalog, pick_variables=["temperature", "salinity"])
    points = dashboard.get_xsection_raster(x_range=(None, None), y_range=(None, None), x=None, y=None)
    assert isinstance(points, hv.Points)
    data = dashboard.data_in_view.collect()
    # two Bornholm missions with 6 profiles each, renumbered consecutively
    assert data["profile_num"].max() == 12
    assert dashboard._stat("max", "temperature") == 15.0

    xlim, ylim, _ = dashboard._scatter_limits()
    assert xlim[0] < xlim[1] and ylim[0] < ylim[1]

    mld = dashboard.mixed_layer_depth("temperature", thresh=0.3, ref_depth=5, verbose=False)
    assert mld.height == 12


def test_layouts_build(catalog):
    dashboard = GliderDashboard(catalog)
    assert isinstance(dashboard.create_dynmap(), pn.Column)
    dashboard.pick_scatter_bool = True
    dashboard.pick_mld = True
    dashboard.pick_contours = "same as above"
    assert isinstance(dashboard.create_dynmap(), pn.Column)
    assert isinstance(dashboard.create_app_instance(), pn.Row)
    assert isinstance(create_metadata_table(catalog), pn.Row)


def test_viewport_uses_time_coverage_of_shown_data(catalog):
    """SEA001's nrt data ends on Jan 20, its delayed data (which is shown) on Jan 25."""
    dashboard = GliderDashboard(catalog, pick_basin="Bornholm Basin")
    x_range = (np.datetime64("2024-01-21"), np.datetime64("2024-01-23"))
    assert list(dashboard.load_viewport_datasets(x_range).index) == ["nrt_SEA001_M1"]


def test_view_without_missions_draws_empty_plot(catalog):
    dashboard = GliderDashboard(catalog, pick_basin="Bornholm Basin")
    x_range = (np.datetime64("2024-02-01"), np.datetime64("2024-02-02"))  # between missions
    points = dashboard.get_xsection_raster(x_range=x_range, y_range=(0, 50), x=np.nan, y=np.nan)
    assert len(points) == 0
    hv.render(rasterize(points, aggregator=dsh.mean("temperature")))  # must not raise
