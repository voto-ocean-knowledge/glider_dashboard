"""Interactive glider data dashboard built with Panel and HoloViews.

:class:`GliderDashboard` holds the user's choices as ``param`` parameters and
renders section plots, scatter plots and profiles of the data in a
:class:`~glider_dashboard.catalog.DataCatalog`. Importing this module does not
load any data; see :mod:`glider_dashboard.app` for how the served app wires a
catalog and a dashboard together.
"""

from __future__ import annotations

import datetime
from importlib import resources

import cmocean
import datashader as dsh
import gsw
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import plotly.express as px
import polars as pl
from holoviews.operation.datashader import rasterize, spread
from holoviews.selection import link_selections
from holoviews.streams import RangeXY, Tap

from . import styles
from .catalog import DataCatalog
from .datasets import MAX_TIME, MIN_TIME, VOTO_INSTITUTION, drop_overlaps_fast
from .processing import concat_missions, mld_profile

#: Hidden widgets get this precedence, visible ones get ``VISIBLE``.
HIDDEN = -10
VISIBLE = 1

SHOW_ALL_INSTITUTIONS = "show all (no filtering)"

#: Panel 1.9 renamed the ``button_type`` option of buttons to ``color``.
BUTTON_COLOR = "color" if "color" in pn.widgets.RadioButtonGroup.param else "button_type"

#: Parameters kept in sync with the page URL, so views can be shared as links.
URL_PARAMETERS = [
    "pick_basin",
    "pick_institution",
    "pick_dsids",
    "pick_toggle",
    "pick_show_ctrls",
    "pick_variables",
    "pick_scatter_bool",
    "pick_scatter_x",
    "pick_scatter_y",
    "pick_scatter",
    "pick_aggregation",
    "pick_mld",
    "pick_cnorm",
    "pick_activate_scatter_link",
    "pick_contours",
    "pick_high_resolution",
    "pick_startX",
    "pick_endX",
    "pick_startY",
    "pick_endY",
    "pick_contour_height",
    "pick_show_decoration",
    "pick_autorange",
    "pick_TS_color_variable",
    "pick_show_metadata",
]

RASTER_TOOLS = [
    "xpan",  # move along
    "xwheel_pan",  # move along x with wheel
    "xwheel_zoom",  # zoom on x with wheel
    "xzoom_in",
    "xzoom_out",
    "crosshair",  # show where the mouse is on axis
    "box_zoom",
    "undo",
    "hover",
    "tap",
    "save",
]


def documentation_markdown() -> str:
    """User guide shown below the plots."""
    return resources.files("glider_dashboard").joinpath("getting_started.md").read_text()


def cbar_range_name(variable: str) -> str:
    """Name of the colorbar range parameter of ``variable``."""
    return f"pick_cbar_range_{variable}"


def dataset_choices(dataset_ids) -> dict[str, str]:
    """Map display labels (without ``nrt_`` prefix) to mission IDs."""
    return {
        dsid.removeprefix("nrt_"): dsid
        for dsid in dataset_ids
        if not dsid.startswith("delayed_")
    }


class GliderDashboard(param.Parameterized):
    """Section plot dashboard for the missions in a :class:`DataCatalog`.

    All ``pick_*`` parameters are user controls. Choices that depend on the
    data (variables, datasets, institutions) are filled in from ``catalog``
    when the dashboard is created.

    Parameters
    ----------
    catalog : DataCatalog
        Data to visualise.
    **params
        Initial parameter values.
    """

    # -- dataset selection -------------------------------------------------- #
    pick_toggle = param.Selector(
        objects=["DatasetID", "SAMBA obs."],
        default="SAMBA obs.",
        label="choose by SAMBA observatory or data ID",
    )
    pick_basin = param.Selector(
        default="Bornholm Basin",
        objects=styles.SAMBA_observatories,
        label="SAMBA observatory",
    )
    pick_institution = param.Selector(
        default=SHOW_ALL_INSTITUTIONS,
        objects=[SHOW_ALL_INSTITUTIONS],
        label="Institution",
        precedence=HIDDEN,
    )
    pick_dsids = param.ListSelector(
        default=[], objects={}, label="DatasetID", precedence=HIDDEN
    )

    # -- section plots ------------------------------------------------------ #
    pick_variables = param.ListSelector(
        default=["temperature"],
        allow_None=False,
        objects=[],
        label="variable",
        doc="Variable used to create colormesh",
    )
    pick_cnorm = param.Selector(
        default="linear",
        objects=["linear", "eq_hist", "log"],
        doc="Colorbar Transformations",
        label="Colourbar Scale",
    )
    pick_autorange = param.Boolean(
        default=True,
        label="autorange colorbars",
        doc="activate manual cbar range controls",
    )
    pick_aggregation = param.Selector(
        default="mean",
        objects=["mean", "std"],
        label="Data Aggregation",
        doc="Method that is applied after binning",
    )
    pick_contours = param.Selector(
        default=None,
        objects=[None, "same as above"],
        label="contour variable",
        doc="Variable presented as contour",
    )
    pick_mld = param.Boolean(default=False, label="MLD", doc="Show Mixed Layer Depth")
    pick_high_resolution = param.Boolean(
        default=False,
        label="Increased Resolution",
        doc="Increases the rendering resolution (slower performance)",
    )

    # -- view range (synced with the URL) ----------------------------------- #
    pick_startX = param.Date(default=None, label="startX", doc="startX")
    pick_endX = param.Date(default=None, label="endX", doc="endX")
    pick_startY = param.Number(default=None, label="startY", doc="startY")
    pick_endY = param.Number(default=None, label="endY", doc="endY")
    pick_contour_height = param.Number(default=None, label="contour_height")

    # -- linked scatter plot ------------------------------------------------ #
    pick_scatter_bool = param.Boolean(
        default=False,
        label="Show scatter diagram",
        doc="Activate scatter diagram",
    )
    pick_scatter = param.Selector(
        default="TS",
        objects=["TS", "profiles", "custom"],
        doc="Type of scatter plot",
        label="scatterplot type",
        precedence=HIDDEN,
    )
    pick_scatter_x = param.Selector(
        default="salinity",
        allow_None=False,
        objects=[],
        label="X-axis variable",
        precedence=HIDDEN,
    )
    pick_scatter_y = param.Selector(
        default="temperature",
        allow_None=False,
        objects=[],
        label="Y-axis variable",
        precedence=HIDDEN,
    )
    pick_TS_color_variable = param.Selector(
        default=None,
        objects=[None],
        label="Colour scatterplot by",
        doc="Colour of the scatterplot",
        precedence=HIDDEN,
    )
    pick_activate_scatter_link = param.Boolean(
        default=False,
        label="enable linked selections",
        doc="enables linked brushing with box and lasso select",
        precedence=HIDDEN,
    )

    # -- page controls ------------------------------------------------------ #
    pick_show_metadata = param.Boolean(default=False, label="show metadata")
    pick_show_ctrls = param.Boolean(default=True, label="show controls")
    pick_show_decoration = param.Boolean(
        default=False, label="Show mission name and start"
    )
    pick_create_download = param.Action(
        lambda x: x.param.trigger("pick_create_download"),
        label="Create download link",
        doc="Create download link",
    )

    #: Fired when a colorbar range slider changes; the colorbar range
    #: parameters are created per variable at runtime.
    refresh_plots = param.Event(precedence=HIDDEN)

    #: Parameters whose valid choices are only known once the catalog is loaded.
    _DATA_DEPENDENT = (
        "pick_variables",
        "pick_scatter_x",
        "pick_scatter_y",
        "pick_TS_color_variable",
        "pick_contours",
        "pick_dsids",
        "pick_institution",
    )

    def __init__(self, catalog: DataCatalog, **params):
        self.catalog = catalog
        self._add_colorbar_range_params(catalog.variables)
        if catalog.has_gdac:
            params.setdefault("pick_toggle", "DatasetID")
        if len(catalog.all_datasets):
            params.setdefault("pick_startX", catalog.all_datasets[MIN_TIME].min())
            params.setdefault("pick_endX", catalog.all_datasets[MAX_TIME].max())
        # set these only after their choices are known, see below
        deferred = {name: params.pop(name) for name in self._DATA_DEPENDENT if name in params}
        super().__init__(**params)

        # state of the current view
        self.data_in_view = None
        self.data_in_view_small = None
        self.visible_datasets = pd.Index([])
        self.stats = pl.DataFrame()
        self.startX = self.endX = self.startY = self.endY = None

        self.markdown = pn.pane.Markdown(documentation_markdown(), width=600, align="center")
        self.profileplots_row = pn.Row()  # filled by `location` when a plot is clicked
        self.lower_control = pn.Row()
        self.mylayout = pn.Row()

        self._set_variable_choices(catalog.variables)
        self.param.pick_dsids.objects = dataset_choices(catalog.mission_ids)
        self.param.pick_institution.objects = [SHOW_ALL_INSTITUTIONS] + list(
            catalog.missions["institution"].unique()
        )
        gdac = catalog.has_gdac
        self.param.pick_institution.precedence = VISIBLE if gdac else HIDDEN
        self.param.pick_dsids.precedence = VISIBLE if gdac else HIDDEN
        self.param.pick_toggle.precedence = HIDDEN if gdac else VISIBLE
        self.param.pick_basin.precedence = HIDDEN if gdac else VISIBLE
        self.param.update(deferred)

        self.param.watch(
            lambda *events: self.param.trigger("refresh_plots"),
            [cbar_range_name(v) for v in catalog.variables],
        )

    @classmethod
    def _add_colorbar_range_params(cls, variables):
        for variable in variables:
            name = cbar_range_name(variable)
            if name not in cls.param:
                cls.param.add_parameter(
                    name,
                    param.Range(
                        default=(0, 1),
                        doc=f"Cbar limits for {variable}",
                        label=variable,
                        precedence=HIDDEN,
                    ),
                )

    def _set_variable_choices(self, variables):
        self.param.pick_variables.objects = variables
        self.param.pick_scatter_x.objects = variables
        self.param.pick_scatter_y.objects = variables
        self.param.pick_TS_color_variable.objects = [None] + variables
        self.param.pick_contours.objects = [None, "same as above"] + variables

    # ------------------------------------------------------------------ #
    # Statistics of the data in view
    # ------------------------------------------------------------------ #

    def _stat(self, statistic: str, column: str):
        """One value from ``self.stats`` (output of ``polars.describe``)."""
        return self.stats.filter(pl.col("statistic") == statistic)[column].item()

    def _padded_range(self, column: str, lower: str = "5%"):
        """Axis limits from the ``lower`` to 99% percentile, padded by 10%."""
        upper = self._stat("99%", column)
        if not isinstance(upper, float):  # e.g. the time column
            return (None, None)
        pad = 0.1 * (upper - self._stat("5%", column))
        return (self._stat(lower, column) - pad, upper + pad)

    def _scatter_limits(self):
        """x, y and colour limits for the scatter plot."""
        if self.data_in_view is None:
            return (None, None), (None, None), (None, None)
        xlim = self._padded_range(self.pick_scatter_x)
        ylim = self._padded_range(self.pick_scatter_y, lower="1%")
        clim = (None, None)
        color = self.pick_TS_color_variable
        if color and isinstance(self._stat("5%", color), float):
            clim = (self._stat("5%", color), self._stat("99%", color))
        return xlim, ylim, clim

    # ------------------------------------------------------------------ #
    # Watchers
    # ------------------------------------------------------------------ #

    def keep_zoom(self, x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    @param.depends("pick_show_ctrls", watch=True)
    def toggle_controls(self):
        if len(self.mylayout):
            self.mylayout[0][0][0].visible = self.pick_show_ctrls

    @param.depends("pick_toggle", "pick_basin", "pick_institution", watch=True)
    def update_datasource(self):
        if self.pick_toggle == "SAMBA obs.":
            self.param.pick_dsids.precedence = HIDDEN
            self.param.pick_basin.precedence = VISIBLE
        elif self.pick_toggle == "DatasetID":
            self.param.pick_dsids.precedence = VISIBLE
            self.param.pick_basin.precedence = HIDDEN
            self.param.pick_institution.precedence = (
                VISIBLE if self.catalog.has_gdac else HIDDEN
            )
            missions = self.catalog.missions
            if self.pick_institution != SHOW_ALL_INSTITUTIONS:
                missions = missions[missions["institution"] == self.pick_institution]
            self.param.pick_dsids.objects = dataset_choices(missions.index)

    @param.depends("pick_basin", "pick_dsids", "pick_toggle", watch=True)
    def change_basin(self):
        self.startX, self.endX = None, None
        self.startY, self.endY = None, None

    @param.depends("pick_autorange", watch=True)
    def preset_clim_slider(self):
        if self.data_in_view is None:
            return
        self.param.update(
            {
                cbar_range_name(variable): (
                    self._stat("1%", variable),
                    self._stat("99%", variable),
                )
                for variable in self.pick_variables
            }
        )

    def location(self, x, y):
        """Show the profile closest to a click on the section plot."""
        if self.data_in_view is None:
            return

        profile_num = (
            self.data_in_view.filter(pl.col("time") > x)
            .first()
            .select(pl.col("profile_num"))
            .collect()[0, 0]
        )
        profile = self.data_in_view.filter(pl.col("profile_num") == profile_num).collect()
        nextprofile = self.data_in_view.filter(
            pl.col("profile_num") == profile_num + 1
        ).collect()

        def profile_curve(profile, variable):
            if "profile_direction" in profile.columns:
                descending = profile.select(pl.col("profile_direction").mean())[0, 0] > 0
            else:  # e.g. GDAC data
                depth = profile["depth"].drop_nulls()
                descending = not depth.first() > depth.last()
            return hv.Curve(
                data=profile.to_pandas().dropna(subset=[variable]),
                kdims=variable,
                vdims="depth",
                label="descending" if descending else "ascending",
            ).opts(
                xlabel=f"{variable.capitalize()} [{styles.units_dict.get(variable, '')}]",
                padding=0.1,
                fontscale=2,
                width=400,
                height=600,
            )

        profile_plots = []
        for variable in self.pick_variables:
            items = [profile_curve(profile, variable)]
            if len(nextprofile) > 0:
                items.append(profile_curve(nextprofile, variable))
            profile_plots.append(
                hv.Overlay(items=items).opts(legend_position="bottom_right", show_legend=True)
            )
        self.profileplots_row.clear()
        self.profileplots_row.append(hv.Layout(profile_plots))

    @param.depends("pick_create_download", watch=True)
    def create_download(self):
        # Not thread safe: all sessions write to the same output.parquet
        self.data_in_view.select(["time", "depth", *self.pick_variables]).sink_parquet(
            "output.parquet"
        )
        self.lower_control.append(
            pn.Row(
                "Download here:",
                pn.widgets.FileDownload(
                    "output.parquet",
                    embed=False,
                    filename="dataframe.parquet",
                    align="end",
                ),
                styles=dict(background="red"),
            )
        )

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #

    def _update_scatter_controls(self):
        """Show the scatter plot controls that apply to the chosen plot type."""
        scatter_params = [
            "pick_scatter",
            "pick_scatter_x",
            "pick_scatter_y",
            "pick_TS_color_variable",
            "pick_activate_scatter_link",
        ]
        if not self.pick_scatter_bool:
            for name in scatter_params:
                self.param[name].precedence = HIDDEN
            return

        self.param.pick_scatter.precedence = VISIBLE
        self.param.pick_TS_color_variable.precedence = VISIBLE
        self.param.pick_activate_scatter_link.precedence = VISIBLE
        if self.pick_scatter == "TS":
            self.pick_scatter_x = "salinity"
            self.pick_scatter_y = "temperature"
            self.param.pick_scatter_x.precedence = HIDDEN
            self.param.pick_scatter_y.precedence = HIDDEN
        elif self.pick_scatter == "profiles":
            self.pick_scatter_y = "depth"
            self.param.pick_scatter_x.precedence = VISIBLE
            self.param.pick_scatter_y.precedence = HIDDEN
        elif self.pick_scatter == "custom":
            self.param.pick_scatter_x.precedence = VISIBLE
            self.param.pick_scatter_y.precedence = VISIBLE

    @param.depends(
        "pick_cnorm",
        "pick_variables",
        "pick_aggregation",
        "pick_mld",
        "pick_basin",
        "pick_institution",
        "pick_dsids",
        "pick_toggle",
        "pick_scatter",
        "pick_scatter_bool",
        "pick_scatter_x",
        "pick_scatter_y",
        "pick_contours",
        "pick_activate_scatter_link",
        "pick_high_resolution",
        "pick_show_decoration",
        "pick_autorange",
        "pick_TS_color_variable",
        "refresh_plots",
    )
    def create_dynmap(self):
        """Build the section plots (and optional scatter plot) for the current choices."""
        self._update_scatter_controls()

        if type(self.pick_startX) is datetime.datetime:
            # User comes in via URL
            self.startX = pd.to_datetime(self.pick_startX)
            self.endX = pd.to_datetime(self.pick_endX)

        range_stream = RangeXY(
            x_range=(self.startX, self.endX), y_range=(self.startY, self.endY)
        )
        range_stream.add_subscriber(self.keep_zoom)
        tap_stream = Tap(x=np.nan, y=np.nan)
        tap_stream.add_subscriber(self.location)

        if len(self.load_viewport_datasets(x_range=(self.startX, self.endX))) == 0:
            self.lower_control.visible = False
            return pn.Column(
                pn.pane.Markdown("""
# No data loaded yet. Please select a DatasetID.

## A list of possible options will be displayed after click into the DatasetID field. You can find the DatasetID field in the 'Choose dataset(s)' menu""")
            )
        self.lower_control.visible = True
        if not self.pick_variables:
            return pn.Column()

        pixel_ratio = 1.0 if self.pick_high_resolution else 0.5
        dmap_raster = hv.DynamicMap(
            self.get_xsection_raster, streams=[range_stream, tap_stream]
        )
        if self.pick_contour_height:
            cheight = int(self.pick_contour_height / len(self.pick_variables))
        else:
            cheight = int((400 + 150 * len(self.pick_variables)) / len(self.pick_variables))

        def rasters(variable):
            aggregator = dsh.std(variable) if self.pick_aggregation == "std" else dsh.mean(variable)
            if self.pick_autorange:
                clim = (None, None)
            else:
                self.param[cbar_range_name(variable)].precedence = VISIBLE
                clim = getattr(self, cbar_range_name(variable), (None, None))
            return rasterize(
                dmap_raster, aggregator=aggregator, pixel_ratio=pixel_ratio
            ).opts(
                invert_yaxis=True,
                colorbar=True,
                clim=clim,
                cmap=styles.cmap_dict.get(variable, cmocean.cm.solar),
                toolbar="above",
                tools=RASTER_TOOLS,
                height=cheight,
                default_tools=[],
                active_tools=["xpan", "xwheel_zoom"],
                responsive=True,
                cnorm=self.pick_cnorm,
                bgcolor="dimgrey",
                clabel=f"{variable.capitalize()}  [{styles.units_dict.get(variable, '')}]",
                ylabel="Depth [m]",
                clim_percentile=bool(self.pick_autorange),
                fontscale=2,
            )

        # one section plot per variable; only the lowest one gets an x-axis
        section_plots = []
        for index, variable in enumerate(self.pick_variables):
            is_last = index == len(self.pick_variables) - 1
            plot = spread(rasters(variable), px=1, how="source")
            if is_last or self.pick_scatter_bool:
                cheight += 40
                plot = plot.opts(
                    labelled=["y"],
                    hooks=[lambda p, _: p.state.update(border_fill_alpha=0)],
                )
            else:
                plot = plot.opts(
                    xaxis=None,
                    hooks=[lambda p, _: p.state.update(border_fill_alpha=0)],
                )
            section_plots.append(plot)

        if self.pick_contours == "same as above":
            plots = hv.Layout([p * hv.operation.contours(p) for p in section_plots])
        elif self.pick_contours is not None:
            contours = hv.operation.contours(rasters(self.pick_contours))
            plots = hv.Layout([p * contours for p in section_plots])
        else:
            plots = hv.Layout(
                [
                    p.opts(xlim=(self.startX, self.endX), ylim=(self.startY, self.endY))
                    for p in section_plots
                ]
            )

        ncols = 1
        if self.pick_scatter_bool:
            scatter = self._create_scatter(range_stream, pixel_ratio)
            if self.pick_activate_scatter_link:
                linked_selection = link_selections.instance()
                plots = linked_selection(plots)
                scatter = linked_selection(scatter)
            scatter = scatter * self._create_scatter_overlay(range_stream, tap_stream)

        if self.pick_show_decoration:
            decorations = hv.DynamicMap(self.get_xsection, streams=[range_stream])
            plots = plots * decorations.opts(ylim=(None, 24))
        if self.pick_mld:
            plots = plots * hv.DynamicMap(self.get_xsection_mld, streams=[range_stream])

        if self.pick_scatter_bool:
            ncols += 1
            plots = plots + scatter.opts(
                padding=(0.05, 0.05), height=cheight, responsive=True, fontscale=2
            )
        return pn.Column(plots.opts(height=cheight).cols(ncols))

    def _create_scatter(self, range_stream, pixel_ratio):
        """Rasterized scatter plot of the data in view."""
        dmap_scatter = hv.DynamicMap(self.get_xsection_TS, streams=[range_stream])
        if self.pick_TS_color_variable:
            scatter = rasterize(
                dmap_scatter,
                pixel_ratio=pixel_ratio,
                aggregator=dsh.mean(self.pick_TS_color_variable),
            ).opts(
                cnorm="linear",
                cmap=styles.cmap_dict.get(self.pick_TS_color_variable, cmocean.cm.solar),
                colorbar=True,
            )
        else:
            scatter = rasterize(dmap_scatter, pixel_ratio=pixel_ratio).opts(cnorm="eq_hist")
        if self.pick_scatter_y in ["depth", "pressure", "z"]:
            scatter = scatter.opts(invert_yaxis=True)

        xlim, ylim, clim = self._scatter_limits()
        return scatter.opts(xlim=xlim, ylim=ylim, clim=clim)

    def _create_scatter_overlay(self, range_stream, tap_stream):
        """Clicked profile (and density contours for TS diagrams) on the scatter plot."""
        clicked_profile = hv.DynamicMap(self.get_xsection_TS_profile, streams=[tap_stream])
        if (self.pick_scatter_x, self.pick_scatter_y) == ("salinity", "temperature"):
            density = hv.DynamicMap(self.get_density_contours, streams=[range_stream])
            clicked_profile = clicked_profile * density.opts(alpha=0.5)
        return clicked_profile

    @param.depends("pick_show_metadata", watch=True)
    def update_markdown(self):
        """
        Updates markdown information below the plots on the page based on the currently visible ranges of data.

        Return
        ------
        all_markdown : str
            String of updated markdown, including paragraph summaries of data within the view and tables
            of statistics.
        """
        if not self.pick_show_metadata:
            self.markdown.object = ""
            return self.markdown

        def stat_time(statistic):
            return self.stats.filter(pl.col("statistic") == statistic)["time"]

        dict_data_summary = {
            "Range Start": stat_time("min").item(),
            "Range End": stat_time("max").item(),
            "Duration (Days)": (
                stat_time("max").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
                - stat_time("min").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
            ).item(),
            "Number of Profiles": self._stat("max", "profile_num")
            - self._stat("min", "profile_num"),
            "Maximum Depth (m)": self._stat("max", "depth"),
        }
        df_data_summary = pd.DataFrame.from_dict(
            dict_data_summary, orient="index", columns=["Value"]
        )

        pl.Config.set_tbl_hide_column_data_types(True)
        pl.Config.set_tbl_hide_dataframe_shape(True)
        table1 = f"""<b>Data Summary</b>
        {df_data_summary.to_html()}"""

        # Table 2: Statistics for the picked variables in "Contour plot options".
        table2 = f"""<b>Picked Variable Statistics</b>
        {self.stats[["statistic"] + self.pick_variables]._repr_html_()}""".replace(
            "&quot;", ""
        )

        # Table 3: Link the metadata for datasetIDs within the current time period.
        def attribute(attributes, name):
            values = attributes.filter(pl.col("Attribute Name") == name).select(pl.col("Value"))
            # IOOS GDAC datasets lack e.g. acknowledgement and creator_url
            return values.item() if len(values) == 1 else ""

        meta_rows = ""
        for datasetid in self.visible_datasets:
            metadata_url = self.catalog.all_datasets.loc[datasetid]["metadata"]
            am = pl.read_csv(metadata_url + ".csv")
            meta_rows += (
                f"<tr><td>{datasetid}</td>"
                f'<td><a href="{metadata_url}.html">link to metadata</a></td>'
                f"<td><a href={attribute(am, 'creator_url')}>{attribute(am, 'institution')}</a></td>"
                f"<td>{attribute(am, 'acknowledgement')}</td></tr>"
            )
        table3 = f"""<b>Datasets in Current Temporal View</b>
<table><tr><th>DatasetID</th><th>Parameters</th><th>Data creator</th><th>Data reference</th></tr>
{meta_rows}
</table>
"""

        all_markdown = f"""
<div style="display: flex; gap: 20px;">
<div style="flex: 1;">
{table1}
</div>
<div style="flex: 1;">
{table2}
</div>
<div style="flex: 1;">
{table3}
</div>
</div>
"""
        self.markdown.object = all_markdown
        return all_markdown

    def load_viewport_datasets(self, x_range) -> pd.DataFrame:
        """Rows of :attr:`DataCatalog.missions` for the missions to show in ``x_range``.

        In "SAMBA obs." mode these are the missions of the chosen observatory
        that overlap the time range (one per start date); in "DatasetID" mode
        the explicitly chosen missions. Also stores the IDs in
        ``self.visible_datasets``.
        """
        x0, x1 = x_range
        if x0 is None:
            # fallback values for initialization
            x0, x1 = np.datetime64("2020-01-01"), np.datetime64("2030-01-01")
        elif not isinstance(x0, np.datetime64):
            x0, x1 = pd.to_datetime(x0), pd.to_datetime(x1)

        if self.pick_toggle == "SAMBA obs.":
            missions = self.catalog.missions
            in_view = missions[
                (missions[MIN_TIME] <= x1)
                & (missions[MAX_TIME] >= x0)
                & (missions["institution"] == VOTO_INSTITUTION)
            ]
            meta = self.catalog.metadata.loc[
                [name for name in in_view.index if name in self.catalog.metadata.index]
            ]
            meta = meta[(meta["basin"] == self.pick_basin) & (meta["project"] == "SAMBA")]
            ids = drop_overlaps_fast(meta).index
        else:
            ids = pd.Index([d for d in self.pick_dsids if d in self.catalog.mission_ids])
        self.visible_datasets = ids
        return self.catalog.missions.loc[ids]

    def get_xsection_mld(self, x_range, y_range):
        dfmld = (
            self.mixed_layer_depth("temperature", thresh=0.3, ref_depth=5, verbose=False)
            .sort("time")
            .to_pandas()
        )
        dfmld["mld"] = dfmld["mld"].rolling(window=5, center=True, min_periods=3).mean()
        return hv.Curve(data=dfmld, kdims="time", vdims="mld")

    def _variables_to_load(self):
        """All variables needed for the current set of plots."""
        variables = list(self.pick_variables)
        if self.pick_contours not in (None, "same as above"):
            variables.append(self.pick_contours)
        if self.pick_scatter_bool:
            # for TS and profile plots
            variables += ["pressure", "salinity", "temperature"]
            variables += [self.pick_scatter_x, self.pick_scatter_y, self.pick_TS_color_variable]
        return list({v for v in variables if v})

    def get_xsection_raster(self, x_range, y_range, x, y):
        """Points of all visible missions; also updates the data in view and its statistics."""
        (x0, x1) = x_range
        (y0, y1) = y_range
        if (x0 is not None) and (x1 is not None):
            # user interacts with the plot; keep the URL in sync (setters have
            # to be in DynamicMap functions to work dynamically)
            self.pick_startX = pd.to_datetime(x0)
            self.pick_endX = pd.to_datetime(x1)
            self.pick_startY = y0
            self.pick_endY = y1
            timedelta = self.pick_endX - self.pick_startX
        else:
            # value doesn't matter, only initialisation run
            timedelta = np.timedelta64(10, "D")

        mission_ids = self.load_viewport_datasets(x_range).index
        variables = self._variables_to_load()
        if len(mission_ids) == 0:
            # e.g. zoomed into a gap between missions: draw nothing and keep
            # the previous data and statistics for the other plots
            empty = pl.DataFrame(
                schema={"time": pl.Datetime("ns"), "depth": pl.Float32}
                | {v: pl.Float32 for v in {*variables, "temperature", "salinity"} - {"time", "depth"}}
            )
            return self.create_single_ds_plot_raster(empty, variables)

        # zoomed out views use the binned data unless high resolution is requested
        use_small = timedelta > np.timedelta64(90, "D") and not self.pick_high_resolution
        frames = [self.catalog.frame(mid, small=use_small) for mid in mission_ids]
        frames_small = [self.catalog.frame(mid, small=True) for mid in mission_ids]

        self.data_in_view = concat_missions(frames).sort("time")
        self.data_in_view_small = (
            concat_missions(frames_small).sort("time")
            if len(mission_ids) > 3
            else self.data_in_view
        )

        missing = set(variables) - set(self.data_in_view.collect_schema().names())
        for element in missing:
            self.data_in_view = self.data_in_view.with_columns(pl.lit(np.nan).alias(element))
            self.data_in_view_small = self.data_in_view_small.with_columns(
                pl.lit(np.nan).alias(element)
            )

        self._set_variable_choices(
            sorted(set(self.data_in_view.collect_schema().names()) | set(variables))
        )

        # statistics of the visible data (expensive)
        if self.startX is not None:
            self.data_in_view_small = self.data_in_view_small.filter(
                (pl.col("time") > self.startX) & (pl.col("time") < self.endX)
            )
        if self.startY is not None:
            self.data_in_view_small = self.data_in_view_small.filter(
                (pl.col("depth") > self.startY) & (pl.col("depth") < self.endY)
            )
        self.stats = self.data_in_view_small.describe((0.01, 0.05, 0.99))
        if self.pick_show_metadata:
            self.update_markdown()
        return self.create_single_ds_plot_raster(self.data_in_view, variables)

    @staticmethod
    def create_single_ds_plot_raster(data, variables):
        """``hv.Points`` over time and depth with ``variables`` as value dimensions."""
        # temperature and salinity always need to be present for the TS lasso to work
        variables = set(variables) | {"temperature", "salinity"}
        kdims = ["time", "depth"]
        # a dimension cannot be both key and value; plot copies instead
        for variable in variables.intersection(kdims):
            data = data.with_columns(pl.col(variable).alias(variable + " "))
            variables.remove(variable)
            variables.add(variable + " ")
        return hv.Points(data=data, kdims=kdims, vdims=list(variables))

    def get_xsection_TS(self, x_range, y_range):
        kdims = [self.pick_scatter_x, self.pick_scatter_y]
        data = self.data_in_view_small
        # the view range is unknown until the user first pans or zooms
        if self.startX is not None:
            data = data.filter((pl.col("time") > self.startX) & (pl.col("time") < self.endX))
        if self.startY is not None:
            data = data.filter((pl.col("depth") > self.startY) & (pl.col("depth") < self.endY))
        if self.pick_TS_color_variable:
            return hv.Points(data=data, kdims=kdims, vdims=[self.pick_TS_color_variable])
        return hv.Points(data=data, kdims=kdims)

    def get_xsection_TS_profile(self, x, y):
        """Highlight the clicked profile in the scatter plot."""
        xvar = self.pick_scatter_x
        yvar = self.pick_scatter_y
        if (self.data_in_view is None) or np.isnan(x) or np.isnan(y):
            return hv.Curve([])

        profile_num = (
            self.data_in_view.filter(pl.col("time") > x)
            .first()
            .select(pl.col("profile_num"))
            .collect()
            .item()
        )
        profile = self.data_in_view.filter(pl.col("profile_num") == profile_num).collect()
        return hv.Curve(
            data=profile.to_pandas().dropna(subset=[xvar, yvar]),
            kdims=xvar,
            vdims=yvar,
        ).opts(color="red", line_width=6)

    def get_density_contours(self, x_range, y_range):
        """Potential density (sigma-0) contours for the TS diagram."""
        # +/- 5 gives plenty of space for the density lines if the user zooms out
        smin = self._stat("1%", "salinity") - 5
        smax = self._stat("99%", "salinity") + 5
        tmin = self._stat("1%", "temperature") - 5
        tmax = self._stat("99%", "temperature") + 5

        xdim = int(round((smax - smin) / 0.1 + 1, 0))
        ydim = int(round((tmax - tmin) + 1, 0))
        ti = np.linspace(1, ydim - 1, ydim) + tmin
        si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
        sigma = gsw.rho(si[np.newaxis, :], ti[:, np.newaxis], 0) - 1000

        return hv.operation.contours(hv.QuadMesh((si, ti, sigma))).opts(
            show_legend=False, cmap="dimgray"
        )

    def get_xsection(self, x_range, y_range):
        """Mission start/end lines and names."""
        meta = self.load_viewport_datasets(x_range)
        if len(meta) == 0:
            return hv.Overlay()
        labels = pd.DataFrame(
            dict(
                time=meta[MIN_TIME].values,
                y=5,
                text=meta.index.str.replace("nrt_", ""),
            )
        )
        return hv.Overlay(
            [
                hv.VLines(meta[MIN_TIME]).opts(color="grey", line_width=1),
                hv.Labels(labels).opts(fontsize=12, text_align="left"),
                hv.VLines(meta[MAX_TIME]).opts(color="grey", line_width=1),
            ]
        )

    def mixed_layer_depth(self, variable, thresh=0.01, ref_depth=-10, verbose=True):
        """
        Calculates the MLD for ungridded glider array.

        You can provide density or temperature.
        The default threshold is set for density (0.01).

        Parameters
        ----------
        variable : str
            variable that will be used for the threshold criteria
        thresh : float
            threshold for difference of variable
        ref_depth : float
            reference depth for difference
        verbose : bool, optional

        Return
        ------
        mld : polars.DataFrame
            One row per profile with columns ``mld`` and ``time``.
        """
        groups = self.data_in_view_small.select(
            pl.col([variable, "depth", "time", "profile_num"])
        ).group_by("profile_num")
        mld = groups.map_groups(
            lambda group_df: mld_profile(group_df, variable, thresh, ref_depth, verbose),
            schema=pl.Schema({"mld": pl.Float32, "time": pl.Time}),
        )
        return mld.collect()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def create_app_instance(self):
        """Build the complete page: controls, plots, profiles and metadata."""
        if pn.state.location:
            # keep the URL in sync with the parameter choices and vice versa
            pn.state.location.sync(self, URL_PARAMETERS)

        self.lower_control = pn.Row(
            pn.Param(
                self,
                parameters=[
                    "pick_show_metadata",
                    "pick_show_ctrls",
                    "pick_show_decoration",
                    "pick_create_download",
                ],
                show_name=False,
                default_layout=pn.Row,
            )
        )
        contentcolumn = pn.Column(
            pn.panel(self.create_dynmap),
            self.lower_control,
            sizing_mode="stretch_width",
        )

        cbar_params = [cbar_range_name(v) for v in self.catalog.variables]
        controls_accordion = pn.Accordion(
            active=[0],  # opening first card by default
            objects=[
                (
                    "Choose dataset(s)",
                    pn.Param(
                        self,
                        parameters=["pick_toggle", "pick_basin", "pick_dsids", "pick_institution"],
                        widgets={
                            "pick_toggle": {
                                "type": pn.widgets.RadioButtonGroup,
                                BUTTON_COLOR: "success",
                            },
                            "pick_dsids": pn.widgets.MultiChoice,
                        },
                    ),
                ),
                (
                    "Section plot options",
                    pn.Param(
                        self,
                        parameters=[
                            "pick_variables",
                            "pick_cnorm",
                            "pick_aggregation",
                            "pick_contours",
                        ],
                        widgets={
                            "pick_variables": pn.widgets.MultiChoice,
                            "pick_cnorm": pn.widgets.RadioButtonGroup,
                            "pick_aggregation": pn.widgets.RadioButtonGroup,
                        },
                    ),
                ),
                (
                    "Linked (scatter-)plots",
                    pn.Param(
                        self,
                        parameters=[
                            "pick_scatter_bool",
                            "pick_scatter",
                            "pick_scatter_x",
                            "pick_scatter_y",
                            "pick_TS_color_variable",
                            "pick_activate_scatter_link",
                        ],
                        widgets={
                            "pick_scatter_bool": pn.widgets.Switch,
                            "pick_scatter": pn.widgets.RadioButtonGroup,
                        },
                    ),
                ),
                (
                    "more",
                    pn.Param(self, parameters=["pick_mld", "pick_high_resolution"]),
                ),
                (
                    "Adjust colorbars",
                    pn.Column(
                        pn.Param(
                            self,
                            parameters=["pick_autorange"] + cbar_params,
                            widgets={p: pn.widgets.EditableRangeSlider for p in cbar_params},
                            show_name=False,
                        ),
                        "upper and lower boundaries can be overwritten manually.",
                    ),
                ),
            ],
        )

        layout = pn.Column(
            pn.Row(controls_accordion, pn.Spacer(width=50), contentcolumn),
            self.profileplots_row,
            self.markdown,
        )
        # Hide the controls only at the very end: hidden controls cannot be
        # accessed above, so hiding earlier would reset URL and manual settings.
        if not self.pick_show_ctrls:
            layout[0][0].visible = False

        self.mylayout.clear()
        self.mylayout.append(layout)
        return self.mylayout


class MetaDashboard(param.Parameterized):
    """Timeline of missions coloured by equipment (currently not shown in the app)."""

    pick_serial = param.Selector(
        default="glider_serial",
        objects=["glider_serial"],
        label="Equipment Ser. No.",
        doc="Track equipment or gliders",
    )

    def __init__(self, catalog: DataCatalog, **params):
        self.catalog = catalog
        super().__init__(**params)
        self.param.pick_serial.objects = [
            "glider_serial",
            "optics_serial",
            "altimeter_serial",
            "irradiance_serial",
            "project",
        ] + list(catalog.all_metadata.columns)

    @param.depends("pick_serial")
    def create_timeline(self):
        dfm = self.catalog.all_metadata.sort_values("basin")
        dims = self.pick_serial
        fig = px.timeline(
            dfm,
            x_start="time_coverage_start (UTC)",
            x_end="time_coverage_end (UTC)",
            y="basin",
            hover_name=dfm.index,
            color_discrete_map={0: "lightgrey", "nan": "grey"},
            hover_data=["ctd_serial", "optics_serial"],
            color=dims,
            pattern_shape=dims,
            height=400,
        )
        fig.update_layout(title=dims, xaxis=dict(rangeslider=dict(visible=True)))
        for d in fig.data:
            d.opacity = 0.7
            d.width = (self.catalog.metadata.deployment_id % 2 + 10) / 12
        fig.layout.autosize = True
        fig.update_layout(height=400)
        return fig


#: Columns of the metadata table page.
METADATA_TABLE_COLUMNS = [
    "datasetID",
    "basin",
    "time_coverage_start (UTC)",
    "time_coverage_end (UTC)",
    "date_issued (UTC)",
    "project",
    "glider_model",
    "glider_serial",
    "ctd_model",
    "ctd_long_name",
    "ctd_serial",
    "oxygen_model",
    "oxygen_serial",
    "oxygen_long_name",
    "oxygen_calibration_date (UTC)",
    "altimeter_model",
    "altimeter_serial",
    "optics_make_model",
    "optics_serial",
    "optics_long_name",
    "irradiance_model",
    "irradiance_serial",
    "turbulence_long_name",
    "metadata_link",
    "summary",
    "comment",
    "variables",
]


def create_metadata_table(catalog: DataCatalog):
    """Filterable table of all VOTO mission metadata."""
    df = catalog.all_metadata.reset_index()
    return pn.Row(
        pn.widgets.Tabulator(
            df[[c for c in METADATA_TABLE_COLUMNS if c in df.columns]],
            frozen_columns=["datasetID"],
            header_filters=True,
            sizing_mode="stretch_both",
            page_size=20,
            disabled=True,  # make cells non-editable
        )
    )
