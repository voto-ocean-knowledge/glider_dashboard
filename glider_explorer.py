import datetime
import logging
import time

import cmocean
import datashader as dsh
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import plotly.express as px
import polars as pl
from holoviews.operation.datashader import (
    rasterize,
    spread,
)
from holoviews.selection import link_selections

# from bokeh.models import DatetimeTickFormatter, HoverTool
from holoviews.streams import (
    RangeXY,
    Tap,
)

import dictionaries
import load_once_data as lod
import utils

# pn.config.reconnect = True
pn.config.notifications = True


def exception_handler(ex):

    logging.error("Error", exc_info=ex)
    # import pdb

    # pdb.set_trace()
    # if (len(GliderDashboard.pick_dsids) == 0) and (
    #    GliderDashboard.pick_toggle == "DatasetID"
    # ):  #
    if GDB.data_in_view is None:
        pn.state.notifications.error(
            "Please proceed by selecting one or more datasets to display",
            duration=10000,
        )
    else:
        pn.state.notifications.error(
            "Please complete/change input parameters", duration=10000
        )
    # import pdb

    # pdb.set_trace()
    # pn.state.notifications.error(f"{ex}")


pn.extension(
    "plotly",
    "mathjax",
    "tabulator",
    throttled=True,
    # sizing_mode="stretch_width",
    template="fast",
    # accent="grey",
    # global_css=[
    #    ":root {--design-primary-color:lightgrey; --design-primary-text-color:black}"
    # ],
    loading_indicator=True,
    exception_handler=exception_handler,
    notifications=True,
    nthreads=0,
    defer_load=True,
)

text_opts = hv.opts.Text(text_align="left", text_color="black", fontsize=10)
ropts = dict(
    toolbar="above",
    tools=["xwheel_zoom", "reset", "xpan", "ywheel_zoom", "ypan"],
    default_tools=[],
    active_tools=["xpan", "xwheel_zoom"],
    bgcolor="dimgrey",
    # ylim=(-8,None)
)
# pn.extension(template="fast")
# pn.extension("plotly")
# pn.extension("tabulator")
# mathjax is currently not used, but could be cool to render latex in markdown
# cudf support works, but is currently not faster


# all_metadata is loaded for the metadata visualisation
# all_metadata, allDatasets = utils.load_metadata()


class GliderDashboard(param.Parameterized):
    pick_display_threshold = param.Number(
        default=1, step=1, bounds=(-10, 10), label="display_treshold"
    )
    pick_variables = param.ListSelector(
        default=["temperature"],
        allow_None=False,
        objects=lod.variables_selectable,
        label="variable",
        doc="Variable used to create colormesh",
        precedence=1,
    )

    pick_GDAC = param.Boolean(
        default=False,
        label="show GDAC data",
        doc="show GDAC data",
        precedence=1,
    )

    pick_scatter_x = param.Selector(
        default="salinity",  # "salinity",
        allow_None=False,
        objects=lod.variables_selectable,
        label="X-axis variable",
        doc="Variable used to create colormesh",
        precedence=-10,
    )

    pick_scatter_y = param.Selector(
        default="temperature",  # "temperature",
        allow_None=False,
        objects=lod.variables_selectable,
        label="Y-axis variable",
        doc="Variable used to create colormesh",
        precedence=-10,
    )
    # show all the basins and all the datasets. I use the nrt data
    # from the metadatatables as keys, so I skip the 'delayed' sets
    # with the lambda function.
    pick_basin = param.Selector(
        default="Bornholm Basin",
        objects=dictionaries.SAMBA_observatories,
        label="SAMBA observatory",
        precedence=1,
    )
    alldslist = list(filter(lambda k: "nrt" in k, lod.dsdict.keys()))
    alldslist = [x for x in alldslist if "_small" not in x]
    if utils.GDAC_data:
        alldslist += list(lod.allDatasetsGDAC.index)
    alldslabels = [
        element[4:] if element[0:4] == "nrt_" else element for element in alldslist
    ]
    objectsdict = dict(zip(alldslabels, alldslist))

    pick_dsids = param.ListSelector(
        default=[],
        objects=objectsdict,
        label="DatasetID",
        precedence=-10,
    )

    pick_toggle = param.Selector(
        objects=["SAMBA obs.", "DatasetID"],
        label="choose by SAMBA observatory or data ID",
    )

    pick_cnorm = param.Selector(
        default="linear",
        objects=["linear", "eq_hist", "log"],
        doc="Colorbar Transformations",
        label="Colourbar Scale",
        precedence=1,
    )

    (locals().update(lod.cbar_range_sliders),)  # noqa

    pick_autorange = param.Boolean(
        default=True,
        label="autorange colorbars",
        doc="activate manual cbar range controls",
        precedence=1,
    )

    # This could be extended with vertical gradient/diff possibly?
    pick_aggregation = param.Selector(
        default="mean",
        objects=["mean", "std"],
        label="Data Aggregation",
        doc="Method that is applied after binning",
        precedence=1,
    )

    pick_aggregation_method = param.Selector(
        default="mean",
        objects=["mean", "min", "max"],  # , "std"],
        label="1D Data Aggregation",
        doc="Method that is applied to aggregate column",
        precedence=1,
    )

    pick_mld = param.Boolean(
        default=False, label="MLD", doc="Show Mixed Layer Depth", precedence=1
    )
    # pick_mean = param.Boolean(
    #    default=False, label="mean", doc="Show column mean", precedence=1
    # )
    pick_startX = param.Date(
        default=lod.allDatasets["minTime (UTC)"].min(),
        label="startX",
        doc="startX",
        precedence=1,
    )
    pick_endX = param.Date(
        default=lod.allDatasets["maxTime (UTC)"].max(),
        label="endX",
        doc="endX",
        precedence=1,
    )
    pick_startY = param.Number(default=None, label="startY", doc="startY", precedence=1)
    pick_endY = param.Number(default=8, label="endY", doc="endY", precedence=1)
    pick_contour_height = param.Number(
        default=None, label="contour_height", precedence=1
    )

    pick_scatter = param.Selector(
        default="TS",
        objects=dict(
            zip(
                ["TS", "profiles", "custom"],
                ["TS", "profiles", "custom"],
            )
        ),
        doc="Type of scatter plot",
        label="scatterplot type",
        precedence=-10,
    )

    pick_scatter_bool = param.Boolean(
        default=False,
        label="Show scatter diagram",
        doc="Activate scatter diagram",
        # precedence=1,
    )

    pick_TS_color_variable = param.Selector(
        default=None,
        objects=([None] + lod.variables_selectable),
        label="Colour scatterplot by",
        doc="Colour of the scatterplot",
        precedence=-10,
    )
    pick_profiles = param.Boolean(
        default=False,
        label="Show profiles",
        doc="Activate profiles diagram",
        precedence=1,
    )
    pick_activate_scatter_link = param.Boolean(
        default=False,
        label="enable linked selections",
        doc="enables linked brushing with box and lasso select",
        precedence=-10,
    )
    button = param.Action(
        lambda x: x.param.trigger("button"),
        # default=False,
        label="Create download link",
        doc="Create download link",
        precedence=1,
    )
    pick_contours = param.Selector(
        default=None,
        objects=(
            [
                None,
                "same as above",
            ]
            + lod.variables_selectable
        ),
        label="contour variable",
        doc="Variable presented as contour",
        precedence=1,
    )
    pick_high_resolution = param.Boolean(
        default=False,
        label="Increased Resolution",
        doc="Increases the rendering resolution (slower performance)",
        precedence=1,
    )
    button_inflow = param.Action(
        lambda x: x.param.trigger("button_inflow"),
        label="Animation event",
        precedence=1,
    )
    pick_show_ctrls = param.Boolean(
        default=True,
        label="show controls",
        precedence=1,
    )
    pick_show_decoration = param.Boolean(
        default=False,
        label="Show mission name and start",
        precedence=1,
    )
    data_in_view = None
    contour_processing = False
    annotations = []
    startX = None
    endX = None
    startY = None
    endY = None

    def update_markdown(self, x_range, y_range):
        p1 = """\
             # About
             Ocean """
        for variable in self.pick_variables:
            description = (
                f"{variable} in [{dictionaries.units_dict.get(variable, '')}], "
            )
            p1 += description
        if self.pick_toggle == "DatasetID":
            p2 = f""" the datasets {self.pick_dsids} """
        else:  # self.pick_toggle == "SAMBA obs.":
            p2 = f"""for the region {self.pick_basin} """
        # try:
        p3 = f"""from {self.data_in_view.select("time").first().collect()[0, 0]} to {self.data_in_view.select("time").last().collect()[0, 0]}. """
        # except:
        #    import pdb
        #    pdb.set_trace()
        p4 = f"""Number of profiles {
            self.data_in_view.select("profile_num").last().collect()[0, 0]
            - self.data_in_view.select("profile_num").first().collect()[0, 0]
        } """

        self.markdown.object = p1 + p2 + p3 + p4  # +r"$$\frac{1}{n}$$"
        return p1 + p2 + p3 + p4

    # empty initialization for use later
    markdown = pn.pane.Markdown("")

    def keep_zoom(self, x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    def create_single_ds_plot_raster(self, data, variables):
        variables = set(variables)
        variables.add("temperature")  # inplace operations
        variables.add("salinity")
        raster = hv.Points(
            data=data,
            kdims=["time", "depth"],
            vdims=list(variables),
            # temp and salinity need to always be present for TS lasso to work, set for unique elements
        )
        return raster

    @param.depends("pick_show_ctrls", watch=True)
    def update_display_threshold(self):
        try:
            # first run, when layout does not exist, this fails deliberately.
            mylayout[0][0][0].visible = self.pick_show_ctrls
        except:
            pass

    @param.depends("pick_toggle", "pick_basin", watch=True)
    def update_datasource(self):
        # toggles visibility
        if not self.pick_GDAC:
            self.param.pick_dsids.objects = set(
                self.param.pick_dsids.objects
            ).intersection(set(lod.allDatasetsVOTO.index))

        if self.pick_toggle == "DatasetID":
            self.param.pick_basin.precedence = -10
            self.param.pick_dsids.precedence = 1
        else:
            self.param.pick_dsids.precedence = -10
            self.param.pick_basin.precedence = 1

    @param.depends("button_inflow", watch=True)
    def execute_event(self):
        self.markdown.object = """\
        # Baltic Inflows
        Baltic Inflows are transporting salt and oxygen into the depth of the Baltic Sea.
        """
        self.startX = np.datetime64("2024-01-15")
        self.endX = np.datetime64("2024-01-18")
        self.pick_startX = np.datetime64("2024-01-15")
        self.pick_endX = np.datetime64("2024-01-18")

        time.sleep(5)
        print("event:plot reloaded")
        text_annotation = hv.Text(
            x=np.datetime64("2024-01-30"),
            y=-20,
            text="Look at this!",
            fontsize=10,
        )
        self.startX = np.datetime64("2024-01-15")
        self.endX = np.datetime64("2024-03-20")
        self.annotations.append(text_annotation)
        self.pick_variables = ["oxygen_concentration"]

        return  # self.dynmap*text_annotation

    @param.depends("pick_basin", "pick_dsids", "pick_toggle", watch=True)
    def change_basin(self):
        # bug: setting watch=True enables correct reset of (y-) coordinates, but leads to double initialization (slow)
        # setting watch=False fixes initialization but does not keep y-coordinate.
        if self.pick_toggle == "SAMBA obs.":
            # first case, , user selected an aggregation, e.g. 'Bornholm Basin'
            meta = lod.metadata[lod.metadata["basin"] == self.pick_basin]
            meta = meta[meta["project"] == "SAMBA"]
            meta = meta[meta["time_coverage_start (UTC)"] > np.datetime64("2021-01-01")]
            meta = utils.drop_overlaps_fast(meta)
            meta = lod.fDs.loc[meta.index]
        else:
            # second case, user selected dids
            meta = lod.allDatasets.loc[self.pick_dsids]  # metadata.loc[self.pick_dsids]

        # hacky way to differentiate if called via synclink or refreshed with UI buttons
        if not len(meta):
            # self.startX = pd.NaT  # None
            # self.endX = pd.NaT  # None
            # self.pick_startX = pd.NaT  # None
            # self.pick_endX = pd.NaT  # None
            return
        incoming_link = not (isinstance(self.pick_startX, pd.Timestamp))
        if not incoming_link:
            mintime = meta["minTime (UTC)"].min()
            maxtime = meta["maxTime (UTC)"].max()
            self.startX, self.endX = None, None  # mintime, maxtime
            self.startY, self.endY = None, None
            # self.pick_startX, self.pick_endX = (mintime, maxtime)
        else:
            self.startX, self.endX = None, None  # mintime, maxtime
            self.startY, self.endY = None, None
            # pass
            # self.pick_startX, self.pick_endX = (self.pick_startX, self.pick_endX)

        # self.startY = None
        # self.endY = 12

    @param.depends(
        "pick_autorange",
        watch=True,
    )
    def preset_clim_slider(self):
        for variable in self.pick_variables:
            if self.data_in_view is not None:
                setattr(
                    self,
                    f"pick_cbar_range_{variable}",
                    (
                        self.stats.loc["1%"][variable],
                        self.stats.loc["99%"][variable],
                    ),
                )

    def location(self, x, y):
        # print(f"Click at {x}, {y}")
        if self.data_in_view is None:
            return None

        profile_num = (
            self.data_in_view.filter(pl.col("time") > x)
            .first()
            .select(pl.col("profile_num"))
        )
        # This way would be more efective than below code
        # profiles = self.data_in_view.filter(
        #    (pl.col("profile_num") == profile_num[0, 0])
        #    | (pl.col("profile_num") == profile_num[0, 0] + 1)
        # ).collect()
        profile = self.data_in_view.filter(
            pl.col("profile_num") == profile_num.collect()[0, 0]
        ).collect()
        nextprofile = self.data_in_view.filter(
            pl.col("profile_num") == profile_num.collect()[0, 0] + 1
        ).collect()
        profile_plots = []

        def create_profile_curve(profile):
            try:
                profilelabel = (
                    "descending"
                    if profile.select(pl.col("profile_direction").mean())[0, 0] > 0
                    else "ascending"
                )
            except:
                df = profile  # .collect()
                profilelabel = (
                    "ascending"
                    if (df["depth"].first() > df["depth"].last())
                    else "descending"
                )
                print("warning, unknown profile direction")
            profilecurve = hv.Curve(
                data=profile.to_pandas().dropna(subset=[variable]),
                kdims=variable,
                vdims="depth",
                label=profilelabel,
            ).opts(
                xlabel=f"{variable} [{dictionaries.units_dict.get(variable, '')}]",
                padding=0.1,
                fontscale=2,
                width=400,
                height=600,
            )
            return profilecurve

        for variable in self.pick_variables:
            items = [create_profile_curve(profile)]
            if len(nextprofile) > 0:
                items.append(create_profile_curve(nextprofile))
            profile_plots.append(
                hv.Overlay(items=items).opts(
                    legend_position="bottom_right", show_legend=True
                )
            )
        self.mylayout[0][2] = pn.Row(hv.Layout(profile_plots))

    @param.depends(
        "button",
        watch=True,
    )
    def create_download(self):
        print("execute create download")
        from io import StringIO

        sio = StringIO()
        self.data_in_view.select(["time", "depth", *self.pick_variables]).sink_parquet(
            "output.parquet"
        )
        # This implementation is not thread save, output is always output.parquet
        self.file_download = pn.widgets.FileDownload(
            "output.parquet", embed=False, filename="dataframe.parquet", align="end"
        )
        # remove previously generate download links
        for index, element in enumerate(mylayout):
            if type(element) == pn.widgets.misc.FileDownload:
                mylayout.pop(index)
        self.mylayout.append(self.file_download)

    @param.depends(
        "pick_dsids",
        "pick_toggle",
        "pick_basin",
        watch=True,
    )
    def update_data(self):
        x_range = (self.startX, self.endX)
        meta, plt_props = self.load_viewport_datasets(x_range)
        metakeys = [
            (
                element.replace("nrt", "delayed")
                if element.replace("nrt", "delayed") in lod.allDatasets.index
                else element
            )
            for element in meta.index
        ]
        varlist = []
        for dsid in metakeys:
            # This is delayed data if available
            if plt_props["zoomed_out"]:
                ds = lod.dsdict[dsid + "_small"]
            else:
                ds = lod.dsdict[dsid]

            # ds = ds.filter(pl.col("profile_num") % plt_props["subsample_freq"] == 0)
            varlist.append(ds)

        # This should only be a temporay hack. I don't want all that data to go into my TS plots.
        # dsconc = utils.voto_concat_datasets2(varlist)
        if varlist:
            dsconc = pl.concat([data for data in varlist], how="diagonal_relaxed")
            dsconc = dsconc.with_columns(pl.col("depth")).sort("time")
            self.param["pick_variables"].objects = dsconc.collect_schema().names()
        # self.data_in_view = dsconc

    @param.depends(
        "pick_cnorm",
        "pick_variables",
        "pick_aggregation",
        "pick_mld",
        # "pick_mean",
        "pick_basin",
        "pick_dsids",
        "pick_toggle",
        "pick_scatter",
        # "pick_TS",
        "pick_scatter_bool",
        "pick_scatter_x",
        "pick_scatter_y",
        "pick_contours",
        "pick_activate_scatter_link",
        "pick_high_resolution",
        # "pick_profiles",
        "pick_display_threshold",
        "pick_show_decoration",  #'pick_startX', 'pick_endX',
        *list(lod.cbar_range_sliders.keys()),  # noqa
        "pick_autorange",
        "pick_TS_color_variable",
        # watch=True,
    )  # outcommenting this means just depend on all, redraw always
    # @pn.io.profile("clustering", engine="snakeviz")
    def create_dynmap(self):
        # self.markdown.object = self.update_markdown()

        # self.startX = self.pick_startX
        # self.endX = self.pick_endX
        # self.startY, self.endY = (self.pick_startY, self.pick_endY)

        # in case coming in over json link
        # self.startX = np.datetime64(self.startX)
        # self.endX = np.datetime64(self.endX)
        #
        # print(self.startX, self.endX, self.startY, self.endY)

        if self.pick_scatter_bool:
            self.param.pick_scatter.precedence = 1
            if self.pick_scatter == "TS":
                self.pick_scatter_x = "salinity"
                self.pick_scatter_y = "temperature"
                self.param.pick_scatter_x.precedence = -10
                self.param.pick_scatter_y.precedence = -10
                self.param.pick_activate_scatter_link.precedence = 1
                self.param.pick_TS_color_variable.precedence = 1
            elif self.pick_scatter == "profiles":
                self.pick_scatter_y = "pressure"
                self.pick_scatter_x = "temperature"
                self.param.pick_scatter_x.precedence = -10
                self.param.pick_scatter_y.precedence = -10
                self.param.pick_TS_color_variable.precedence = 1
                self.param.pick_activate_scatter_link = 1
            elif self.pick_scatter == "custom":
                self.pick_scatter_x = self.pick_scatter_x
                self.pick_scatter_y = self.pick_scatter_y
                self.param.pick_scatter_x.precedence = 1
                self.param.pick_scatter_y.precedence = 1
                self.param.pick_TS_color_variable.precedence = 1
                self.param.pick_activate_scatter_link.precedence = 1

        else:
            self.param.pick_scatter.precedence = -10
            self.param.pick_scatter_x.precedence = -10
            self.param.pick_scatter_y.precedence = -10
            self.param.pick_TS_color_variable.precedence = -10
            self.param.pick_activate_scatter_link.precedence = -10

        # commonheights = 1000

        if type(self.pick_startX) is datetime.datetime:
            # User comes in via URL
            self.startX = pd.to_datetime(self.pick_startX)
            self.endX = pd.to_datetime(self.pick_endX)

        range_stream = RangeXY(
            x_range=(self.startX, self.endX), y_range=(self.startY, self.endY)
        )  # x_range=x_range, y_range=y_range).rename()
        range_stream.add_subscriber(self.keep_zoom)
        # range_stream.add_subscriber(self.update_markdown) # Is always one step after, thus deactivated here

        # Create a callback for a dynamic map
        tap_stream = Tap(x=np.nan, y=np.nan)
        tap_stream.add_subscriber(self.location)

        pick_cnorm = "linear"

        # print(
        #    self.load_viewport_datasets(x_range=(self.startX, self.endX)),
        #    not self.load_viewport_datasets(x_range=(self.startX, self.endX)),
        # )
        # import pdb

        # pdb.set_trace()
        if len(self.load_viewport_datasets(x_range=(self.startX, self.endX))[0]) == 0:
            return pn.Column(
                "# Please select a DatasetID. A list of possible options will be displayed after click into the DatasetID field."
            )
        dmap_raster = hv.DynamicMap(
            self.get_xsection_raster,
            streams=[range_stream, tap_stream],
        )  # .opts(framewise=True)

        if self.pick_high_resolution:
            pixel_ratio = 1.0
        else:
            pixel_ratio = 0.5

        if self.pick_scatter_bool:
            dmap_TS = hv.DynamicMap(
                self.get_xsection_TS,
                streams=[range_stream],
                # cache_size=1,
            )

            if not self.pick_TS_color_variable:
                dmapTSr = rasterize(
                    dmap_TS,
                    pixel_ratio=pixel_ratio,
                ).opts(
                    cnorm="eq_hist",
                )
            else:
                dmapTSr = rasterize(
                    dmap_TS,
                    pixel_ratio=pixel_ratio,
                    aggregator=dsh.mean(self.pick_TS_color_variable),
                ).opts(
                    cnorm="linear",
                    cmap=dictionaries.cmap_dict.get(
                        self.pick_TS_color_variable, cmocean.cm.solar
                    ),
                    # clabel=f"{self.pick_variable}  [{dictionaries.units_dict[self.pick_variable]}]",
                    colorbar=True,
                )
            if self.pick_scatter_y in ["depth", "pressure", "z"]:
                dmapTSr = dmapTSr.opts(invert_yaxis=True)

        """
            dcont = hv.DynamicMap(
                self.get_density_contours, streams=[range_stream]
            ).opts(
                alpha=0.5,
            )
        """

        dmap_decorators = hv.DynamicMap(
            self.get_xsection,
            streams=[range_stream],  # cache_size=1
        )
        if self.pick_mld:
            # Important!!! Compute MLD only once and apply it to all plots!!!
            dmap_mld = hv.DynamicMap(
                self.get_xsection_mld,
                streams=[range_stream],  # cache_size=1
            )  # .opts(responsive=True)

        plots_dict = dict(dmap_rasterized=dict(), dmap_rasterized_contour=dict())
        if len(self.pick_variables):
            if self.pick_contour_height:
                cheight = int(self.pick_contour_height / len(self.pick_variables))
            else:
                cheight = int(
                    (400 + 150 * len(self.pick_variables)) / len(self.pick_variables)
                )
        else:
            cheight = 0

        # make sure all range sliders for non-activated variables are hidden:
        # for variable in list(set(variables_selectable).difference(self.pick_variables)):
        #    self.param[f"pick_cbar_range_{variable}"].precedence = -10
        # self.param[f"pick_autorange_{variable}"].precedence = -10

        # variables = self.pick_variables
        def rasters(variable):
            if self.pick_aggregation == "mean":
                means = dsh.mean(variable)
            if self.pick_aggregation == "std":
                means = dsh.std(variable)
            if self.pick_autorange:
                clim = (None, None)
            #    clim = (self.stats[variable].loc["1%"], self.stats[variable].loc["99%"])

            else:
                self.param[f"pick_cbar_range_{variable}"].precedence = 1
                clim = eval(f"self.pick_cbar_range_{variable}")

            mraster = rasterize(
                dmap_raster,
                aggregator=means,
                # x_sampling=8.64e13/48,
                # y_sampling=0.2,
                pixel_ratio=pixel_ratio,
                # robust=True if self.pick_autorange else False,
            ).opts(
                invert_yaxis=True,  # Would like to activate this, but breaks the hover tool
                colorbar=True,
                # clim_percentile=clim_percentile,
                clim=clim,
                cmap=dictionaries.cmap_dict.get(variable, cmocean.cm.solar),
                toolbar="above",
                tools=[
                    "xpan",  # move along
                    "xwheel_pan",  # move along x with wheel
                    "xwheel_zoom",  # zoom on x with wheel
                    "xzoom_in",  # zoom in on x
                    "xzoom_out",  # zoom out on x
                    "crosshair",  # show where the mouse is on axis
                    "box_zoom",  # zoom on selection along x
                    "undo",  # undo action
                    "hover",
                    "tap",
                    "save",
                    # "redo",
                ],
                height=cheight,
                default_tools=[],
                active_tools=["xpan", "xwheel_zoom"],
                # default_tools=[],
                responsive=True,  # this currently breaks when activated with MLD
                # width=800,
                # int(500/(len(self.pick_variables))),#250+int(250*2/len(self.pick_variables)), #500, 250,
                cnorm=self.pick_cnorm,
                bgcolor="dimgrey",
                clabel=f"{variable}  [{dictionaries.units_dict.get(variable, '')}]",  # self.pick_variable,pick_TS_col
                clim_percentile=True if self.pick_autorange else False,
                fontscale=2,
                # framewise=True,
            )

            return mraster

        for index, variable in enumerate(self.pick_variables):
            if (
                (index < len(self.pick_variables) - 1)
                and not self.pick_scatter_bool  # _bool
                and not self.pick_profiles
            ):
                plots_dict["dmap_rasterized"][variable] = spread(
                    rasters(variable),
                    px=1,
                    how="source",  # , shape="circle"
                ).opts(
                    # ylim=(self.startY, self.endY),
                    # xlim=(self.startX, self.endX),
                    xaxis=None,
                    hooks=[lambda p, _: p.state.update(border_fill_alpha=0)],
                )
            else:
                cheight += 50
                plots_dict["dmap_rasterized"][variable] = spread(
                    rasters(variable),
                    px=1,
                    how="source",  # , shape="circle"
                ).opts(
                    # ylim=(self.startY, self.endY),
                    # xlim=(self.startX, self.endX),
                    hooks=[lambda p, _: p.state.update(border_fill_alpha=0)],
                )
            # if self.pick_show_decoration:
            #    plots_dict["dmap_rasterized"][variable] = plots_dict["dmap_rasterized"][
            #        variable
            # ].opts(ylim=(None, 12))
        if (self.pick_contours is not None) and (self.pick_contours != "same as above"):
            plots_dict["dmap_rasterized_contour"] = rasters(self.pick_contours)

        if self.pick_contours:
            if self.pick_contours == "same as above":
                contourplots = hv.Layout(
                    [
                        element * hv.operation.contours(element)
                        for element in plots_dict["dmap_rasterized"].values()
                    ]
                )
            else:
                overlay_contours = hv.operation.contours(
                    plots_dict["dmap_rasterized_contour"]
                )
                contourplots = hv.Layout(
                    [
                        element * overlay_contours
                        for element in plots_dict["dmap_rasterized"].values()
                    ]
                )
        else:
            # There is nothing to show here, return empty
            if self.pick_variables:
                contourplots = hv.Layout(
                    [
                        element.opts(xlim=(self.startX, self.endX))
                        for element in plots_dict["dmap_rasterized"].values()
                    ]
                )
            else:
                return pn.Column()
        ncols = 1
        if self.pick_scatter_bool:  # or self.pick_profiles:
            # link the contourplots with the scatterplot
            mpg_ls = link_selections.instance()
            if self.pick_activate_scatter_link:
                contourplots = mpg_ls(contourplots)

            if self.pick_scatter_bool:
                if self.data_in_view is not None:
                    if isinstance(self.stats.loc["99%"][self.pick_scatter_x], float):
                        diffx = (
                            self.stats.loc["99%"][self.pick_scatter_x]
                            - self.stats.loc["5%"][self.pick_scatter_x]
                        )

                        xlim = (
                            self.stats.loc["5%"][self.pick_scatter_x] - 0.1 * diffx,
                            self.stats.loc["99%"][self.pick_scatter_x] + 0.1 * diffx,
                        )
                    else:
                        # for example time variable
                        xlim = (None, None)
                    if isinstance(self.stats.loc["99%"][self.pick_scatter_x], float):
                        diffy = (
                            self.stats.loc["99%"][self.pick_scatter_y]
                            - self.stats.loc["5%"][self.pick_scatter_y]
                        )
                        ylim = (
                            self.stats.loc["1%"][self.pick_scatter_y] - 0.1 * diffy,
                            self.stats.loc["99%"][self.pick_scatter_y] + 0.1 * diffy,
                        )
                    else:
                        ylim = (None, None)
                else:
                    xlim = ylim = (None, None)
                if self.pick_TS_color_variable:
                    clim = (
                        self.stats.loc["5%"][self.pick_TS_color_variable],
                        self.stats.loc["99%"][self.pick_TS_color_variable],
                    )
                else:
                    clim = (None, None)
                ncols += 1
                if self.pick_activate_scatter_link:
                    dmapTSr = mpg_ls(
                        dmapTSr.opts(xlim=xlim, ylim=ylim, clim=clim)
                    )  # * dcont
                else:
                    dmapTSr = dmapTSr.opts(xlim=xlim, ylim=ylim, clim=clim)  # * dcont
            if self.pick_profiles:
                ncols += 1
                if self.pick_activate_scatter_link:
                    dmap_profilesr = mpg_ls(dmap_profilesr)  # mpg_ls(dmap_profilesr)
                else:
                    dmap_profilesr = dmap_profilesr

        # annotations are currently broken, fix here
        # for annotation in self.annotations:
        #    print("insert text annotations defined in events")
        #    self.dynmap = self.dynmap * annotation#
        #
        #    return linked_plots

        if self.pick_show_decoration:
            contourplots = contourplots * dmap_decorators.opts(ylim=(None, 24))
        contourplots = contourplots * dmap_mld if self.pick_mld else contourplots
        contourplots = (
            (
                (contourplots)
                + dmapTSr.opts(
                    padding=(0.05, 0.05),
                    height=cheight,
                    responsive=True,
                    fontscale=2,
                )
            )
            if self.pick_scatter_bool
            else contourplots
        )
        contourplots = (
            (contourplots)
            + dmap_profilesr.opts(
                height=cheight,
                responsive=True,
                fontscale=2,
            )
            if self.pick_profiles
            else contourplots
        )
        contourplots = contourplots  # .redim.range(
        # time=(self.startX, self.endX), depth=(self.startY, self.endY)
        # )
        # ToDo: Test if this actually helps the garbage collector
        # self.stats = None
        # self.data_in_view = None
        # self.data_in_view_small = None

        return pn.Column(contourplots.opts(height=cheight).cols(ncols))

    def create_mean(self):
        self.startX = self.pick_startX
        self.endX = self.pick_endX

        # in case coming in over json link
        self.startX = np.datetime64(self.startX)
        self.endX = np.datetime64(self.endX)
        x_range = (self.startX, self.endX)
        y_range = (self.startY, self.endY)
        range_stream = RangeXY(x_range=x_range, y_range=y_range).rename()
        dmap = hv.DynamicMap(
            self.get_xsection,
            streams=[range_stream],
            # cache_size=1,
        )
        dmap_mean = (
            hv.DynamicMap(
                self.get_xsection_mean,
                streams=[range_stream],  # cache_size=1
            ).opts(
                # invert_yaxis=True, # Would like to activate this, but breaks the hover tool
                # colorbar=True,
                # cmap=dictionaries.cmap_dict[self.pick_variable],
                # toolbar="above",
                color="black",
                tools=["xwheel_zoom", "reset", "xpan"],
                default_tools=[],
                # responsive=True, # this currently breaks when activated with MLD
                # width=800,
                # height=commonheights,
                # cnorm=self.pick_cnorm,
                active_tools=["xpan", "xwheel_zoom"],
                bgcolor="dimgrey",
                # clabel=self.pick_variable,
            )
            * dmap
        )  # .opts(responsive=True)

        return dmap_mean

    def load_viewport_datasets(self, x_range):
        """
        Returns a pandas dataframe containing keys of the datasets that are in the current view.
        This is currently based on the metadata information "time_coverage_start/end (UTC), but should
        be generalized to minTime (UTC) to be compatible with the allDatasets table instead of metadata tables.
        """
        (x0, x1) = x_range
        # dt = x1 - x0
        # dtns = dt / np.timedelta64(1, "ns")
        plt_props = {}
        if x0 is None:
            # fallback values for initialization
            x0 = np.datetime64("2020-01-01")
            x1 = np.datetime64("2030-01-01")
        elif type(x_range[0]) == np.datetime64:
            x0 = x_range[0]
            x1 = x_range[1]
        else:
            x0 = pd.to_datetime(x_range[0])
            x1 = pd.to_datetime(x_range[1])

        fDs = lod.allDatasets.loc[
            [name for name in lod.all_dataset_names if "_small" not in name]
        ]
        if (x0 is None) or (x1 is None):  # or (np.isnan(x0)) or (np.isnan(x1)):
            fD_inview = fDs
        else:
            fD_inview = fDs[
                # x0 and x1 are the time start and end of our view, the other times
                # are the start and end of the individual datasets. To increase
                # perfomance, datasets are loaded only if visible, so if
                # 1. it starts within our view...
                ((fDs["minTime (UTC)"] >= x0) & (fDs["minTime (UTC)"] <= x1))
                |
                # 2. it ends within our view...
                ((fDs["maxTime (UTC)"] >= x0) & (fDs["maxTime (UTC)"] <= x1))
                |
                # 3. it starts before and ends after our view (zoomed in)...
                ((fDs["minTime (UTC)"] <= x0) & (fDs["maxTime (UTC)"] >= x1))
                |
                # 4. or it both, starts and ends within our view (zoomed out)...
                ((fDs["minTime (UTC)"] >= x0) & (fDs["maxTime (UTC)"] <= x1))
            ]

        # fD_inview = fDs

        # print(fD_inview)
        # mydslist = [name for name in all_dataset_names if '_small' not in name]

        if self.pick_toggle == "SAMBA obs.":
            # first case, , user selected an aggregation, e.g. 'Bornholm Basin'
            #
            fD_inview = fD_inview[
                fD_inview["institution"] == "Voice of the Ocean Foundation"
            ]
            meta = lod.metadata.loc[
                [name for name in fD_inview.index if "delayed" not in name]
            ]
            meta = meta[meta["basin"] == self.pick_basin]
            meta = meta[meta["project"] == "SAMBA"]
            meta = utils.drop_overlaps_fast(meta)

        else:
            meta = lod.allDatasets.loc[self.pick_dsids]

        # print(f'len of meta is {len(meta)} in load_viewport_datasets')
        if (x1 - x0) > np.timedelta64(720, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = True
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 25
        elif (x1 - x0) > np.timedelta64(360, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = True
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 10  # 25  # 10
        elif (x1 - x0) > np.timedelta64(180, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 6
        elif (x1 - x0) > np.timedelta64(90, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 2
        else:
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 10
            plt_props["subsample_freq"] = 1
        return lod.allDatasets.loc[meta.index], plt_props

    def get_xsection_mld(self, x_range, y_range):
        # print("DATA IN VIEW:", self.data_in_view.collect())
        dfmld = (
            self.mixed_layer_depth(
                "temperature",
                thresh=0.3,
                verbose=False,
                ref_depth=5,
            )
        ).sort("time")

        dfmld = dfmld.to_pandas()
        dfmld["mld"] = dfmld["mld"].rolling(window=5, center=True, min_periods=3).mean()
        # dfmld["mld2"] = (
        #    -dfmld["mld"].rolling(window=5, center=True, min_periods=3).mean()
        # )

        """
        mldscatter = dfmld.hvplot.line(
            x="time",
            y="mld",
            color="white",
            alpha=0.5,
            responsive=True,
        )"""

        # print(dfmld)
        mldscatter = hv.Curve(data=dfmld, kdims="time", vdims="mld")
        # mldscatter2 = hv.Curve(data=dfmld, kdims="time", vdims="mld2")

        return mldscatter  # * mldscatter2

    def get_xsection_mean(self, x_range, y_range):
        # This method is not adapted for multiple variables (pick_variables) yet
        try:
            dscopy = utils.add_dive_column(self.data_in_view).compute()
        except:
            dscopy = utils.add_dive_column(self.data_in_view)
        # dscopy["depth"] = -dscopy["depth"]
        # mld = gt.physics.mixed_layer_depth(
        #    dscopy.to_xarray(), "temperature", thresh=0.3, verbose=True, ref_depth=5
        # )
        if self.pick_aggregation_method == "mean":
            groups = (
                dscopy.reset_index()[["time", self.pick_variable, "profile_num"]]
                .groupby(by="profile_num")
                .mean()
            )  # .time
        elif self.pick_aggregation_method == "max":
            groups = (
                dscopy.reset_index()[["time", self.pick_variable, "profile_num"]]
                .groupby(by="profile_num")
                .max()
            )  # .time
        elif self.pick_aggregation_method == "min":
            groups = (
                dscopy.reset_index()[["time", self.pick_variable, "profile_num"]]
                .groupby(by="profile_num")
                .min()
            )  # .time
        # elif self.pick_aggregation_method == 'std':
        #    groups = dscopy.reset_index()[['time', self.pick_variable, 'profile_num']].groupby(by="profile_num").std()#.time

        gtime = groups.time
        gmean = groups[self.pick_variable]
        # gtmean = dscopy.reset_index().groupby(by="profile_num")[self.pick_variable].mean()
        # mld=-mld.rolling(10, center=True).mean().values
        dfmean = (
            pd.DataFrame.from_dict(dict(time=gtime.values, mean=gmean.values))
            .sort_values(by="time")
            .dropna()
        )  # .rolling(window=4).mean()
        dfmean["mean"] = dfmean["mean"].rolling(4, center=True).mean().values

        meanline = dfmean.hvplot.line(
            x="time",
            y="mean",
            responsive=True,
        )

        return meanline

    def get_xsection_raster(self, x_range, y_range, x, y):
        (x0, x1) = x_range
        if (x0 is not None) and (x1 is not None):
            # user interacts with program via ui, the URL is dynamically updated to keep up to date
            # setters have to be in DynamicMap functions to work dynamically!)
            self.pick_startX = pd.to_datetime(x0)
            self.pick_endX = pd.to_datetime(x1)

        meta, plt_props = self.load_viewport_datasets(x_range)
        metakeys = [
            (
                element.replace("nrt", "delayed")
                if element.replace("nrt", "delayed") in lod.allDatasetsVOTO.index
                else element
            )
            for element in meta.index
        ]

        #################################################################
        # This is currently hard to understand, but:                    #
        # varlist are the datasets that are visualized, either the      #
        # original .parquet files or the _small.parquet version if      #
        # zoomed out                                                    #
        # varlist_small are the nrt_*.parquet files, that are evaluated #
        # to create statistics (quantiles for data ranges)              #
        #################################################################

        if (self.pick_contours is not None) and (self.pick_contours != "same as above"):
            variables = self.pick_variables + [self.pick_contours]
        else:
            variables = self.pick_variables
        if self.pick_scatter_bool:  # == True:
            variables = variables + [
                "pressure",
                "salinity",
                "temperature",
            ]  # for TS and profile plots
            if self.pick_scatter_x:
                variables = variables + [self.pick_scatter_x]
            if self.pick_scatter_y:
                variables = variables + [self.pick_scatter_y]
            if self.pick_TS_color_variable:
                variables = variables + [self.pick_TS_color_variable]
        variables = list(set(variables))
        varlist = []
        varlist_small = []

        for dsid in metakeys:
            # This is delayed data if available
            if plt_props["zoomed_out"] and (not self.pick_high_resolution):
                ds = lod.dsdict[dsid + "_small"]
            else:
                ds = lod.dsdict[dsid]  # + "_small"]

            # ds = ds.filter(pl.col("profile_num") % plt_props["subsample_freq"] == 0)
            varlist.append(ds)

        for dsid in meta.index:
            # This is only the nrt data
            ds = lod.dsdict[dsid]
            varlist_small.append(ds)

        if self.pick_scatter_bool:  # _bool:  # or self.pick_profiles:
            nanosecond_iterator = 1
            for ndataset in varlist:
                ndataset = ndataset.with_columns(
                    pl.col("time") + np.timedelta64(nanosecond_iterator, "ns")
                )
                nanosecond_iterator += 1
            for ndataset in varlist_small:
                ndataset = ndataset.with_columns(
                    pl.col("time") + np.timedelta64(nanosecond_iterator, "ns")
                )
                nanosecond_iterator += 1

        # This should only be a temporay hack. I don't want all that data to go into my TS plots.
        dsconc = utils.voto_concat_datasets2(varlist)
        dsconc = dsconc.with_columns(pl.col("depth")).sort("time")

        dsconc_small = utils.voto_concat_datasets2(varlist_small)
        dsconc_small = dsconc_small.with_columns(pl.col("depth")).sort("time")

        self.data_in_view = dsconc
        self.data_in_view_small = dsconc_small
        """ WILL I NEED THIS FOR MLD COMPUTATION? """
        # if self.startX is not None:
        """
        self.data_in_view = dsconc.filter(
            (pl.col("time") > self.startX) & (pl.col("time") < self.endX)
            # & (pl.col("depth") > self.startY)
            # & (pl.col("depth") < self.endY)
        )  # .dropna(subset=['temperature', 'salinity'])
        self.data_in_view_small = dsconc_small.filter(
            (pl.col("time") > self.startX) & (pl.col("time") < self.endX)
            # & (pl.col("depth") > self.startY)
            # & (pl.col("depth") < self.endY)
        )
        # else:

        print(
            "The length of the datasets is:",
            dsconc.select(pl.len()).collect().item(),
            dsconc_small.select(pl.len()).collect().item(),
        )
        """
        # THIS IS EXPENSIVE. I SHOULD CREATE STATS ONLY WHERE NEEDED; ESPECIALLY WITH .to_pandas()
        self.stats = (
            self.data_in_view_small.describe(  # .select(variables)  # .select(variables)  # .select(pl.col(self.pick_variables))
                (0.01, 0.05, 0.99)
            )
            .to_pandas()
            .set_index("statistic")
        )
        self.update_markdown(x_range, y_range)
        mplt = self.create_single_ds_plot_raster(
            data=self.data_in_view, variables=variables
        )
        # print(self.startX, self.endX, self.startY, self.endY)

        return mplt

    def get_xsection_TS(self, x_range, y_range):
        vdims = ["depth", "time"]
        if self.pick_TS_color_variable:
            vdims.append(self.pick_TS_color_variable)
        mplt = hv.Points(
            data=self.data_in_view.filter(
                (pl.col("time") > self.startX)
                & (pl.col("time") < self.endX)
                & (pl.col("depth") > self.startY)
                & (pl.col("depth") < self.endY)
            ),
            kdims=[self.pick_scatter_x, self.pick_scatter_y],
            vdims=vdims,
            # temp and salinity need to always be present for TS lasso to work, set for unique elements
        )

        return mplt

    def get_xsection_profiles(self, x_range, y_range):
        low = self.stats.loc["1%"][self.pick_variables[0]]
        high = self.stats.loc["99%"][self.pick_variables[0]]

        mplt = hv.Points(
            data=self.data_in_view, kdims=[self.pick_variables[0], "depth"]
        ).opts(
            xlim=(low * 0.95, high * 1.05),
        )
        return mplt

    def get_density_contours(self, x_range, y_range):
        import gsw
        # +/- 5 gives plently of space for the density line drawing, if user zoomes out.

        smin, smax = (
            self.stats.loc["5%"]["salinity"] - 5,
            self.stats.loc["99%"]["salinity"] + 5,
        )
        tmin, tmax = (
            self.stats.loc["5%"]["temperature"] - 5,
            self.stats.loc["99%"]["temperature"] + 5,
        )

        xdim = round((smax - smin) / 0.1 + 1, 0)
        ydim = round((tmax - tmin) + 1, 0)

        # Create empty grid of zeros
        dens = np.zeros((int(ydim), int(xdim)))

        # Create temp and salt vectors of appropiate dimensions
        ti = np.linspace(1, ydim - 1, int(ydim)) + tmin
        si = np.linspace(1, xdim - 1, int(xdim)) * 0.1 + smin

        # Loop to fill in grid with densities
        for j in range(0, int(ydim)):
            for i in range(0, int(xdim)):
                dens[j, i] = gsw.rho(si[i], ti[j], 0)

        # Substract 1000 to convert to sigma-t
        dens = dens - 1000

        dcont = hv.QuadMesh((si, ti, dens))
        dcont = hv.operation.contours(
            dcont,
        ).opts(
            show_legend=False,
            cmap="dimgray",
            xlim=(
                self.stats.loc["5%"]["salinity"]
                - 1,  # 5% because we get 0PSU readings at surface.
                self.stats.loc["99%"]["salinity"] + 1,
            ),
            ylim=(
                self.stats.loc["1%"]["temperature"] - 1,
                self.stats.loc["99%"]["temperature"] + 1,
            ),
        )
        return dcont

    def create_None_element(self, type):
        # This is just a hack because I can't return None to dynamic maps
        if type == "Overlay":
            element = hv.Overlay(
                hv.HLine(0).opts(color="black", alpha=0.1)
                * hv.HLine(0).opts(color="black", alpha=0.1)
            )
        elif type == "Spikes":
            element = hv.Spikes().opts(color="black", alpha=0.1)
        return element

    def get_xsection(self, x_range, y_range):
        """
        (x0, x1) = x_range
        try:
            # necessary if changing dsids dynamically
            x0 = x0.to_datetime64()
            x1 = x1.to_datetime64()
        except:
            pass
        t1 = time.perf_counter()
        """
        meta, plt_props = self.load_viewport_datasets(x_range)
        """
        try:
            meta_start_in_view = meta[(meta["minTime (UTC)"] > x0)]
            meta_end_in_view = meta[(meta["maxTime (UTC)"] < x1)]
        except:
            import pdb

            pdb.set_trace()
        """
        startvlines = (
            hv.VLines(meta["minTime (UTC)"]).opts(
                color="grey", line_width=1
            )  # , spike_length=20)
            # .opts(position=-10)
        )
        endvlines = (
            hv.VLines(meta["maxTime (UTC)"]).opts(
                color="grey", line_width=1
            )  # , spike_length=20)
            # .opts(position=-10)
        )

        data = pd.DataFrame.from_dict(
            dict(
                time=meta["minTime (UTC)"].values,
                y=5,
                text=meta.index.str.replace("nrt_", ""),
            )
        )
        ds_labels = hv.Labels(data).opts(
            fontsize=12,
            text_align="left",  # plt_props['dynfontsize'],
        )
        plotslist = []
        if len(meta) > 0:
            plotslist.append(startvlines)
            plotslist.append(ds_labels)
        if len(meta) > 0:
            plotslist.append(endvlines)
        if plotslist:
            return hv.Overlay(plotslist)  # reduce(lambda x, y: x*y, plotslist)
        else:
            return hv.Overlay()  # return self.create_None_element("Overlay")

    def mixed_layer_depth(self, variable, thresh=0.01, ref_depth=-10, verbose=True):
        """
        Calculates the MLD for ungridded glider array.

        You can provide density or temperature.
        The default threshold is set for density (0.01).

        Parameters
        ----------
        ds : xarray.Dataset Glider dataset
        variable : str
            variable that will be used for the threshold criteria
        thresh : float=0.01 threshold for difference of variable
        ref_depth : float=10 reference depth for difference
        return_as_mask : bool, optional
        verbose : bool, optional

        Return
        ------
        mld : array
            will be an array of depths the length of the
            number of unique dives.
        """

        groups = self.data_in_view_small.select(
            pl.col([variable, "depth", "time", "profile_num"])
        ).group_by("profile_num")

        mld = groups.map_groups(
            lambda group_df: mld_profile(group_df, "temperature", 0.3, 5, False),
            schema=pl.Schema(
                {
                    "mld": pl.Float32,
                    "time": pl.Time,  # ("us"),
                }
            ),
        )
        return mld.collect()

    # @param.depends("pick_autorange")
    def create_app_instance(self):

        def create_column(hex_id=None):
            """
            Dynamically add a new row to the app.
            """
            global meancolumn
            meancolumn = pn.Column(
                self.create_mean(),
                height=500,
            )
            contentcolumn.append(meancolumn)
            contentcolumn.height = 1050

        def remove_column(hex_id=None):
            """
            Dynamically remove a column from the app.
            """
            contentcolumn.remove(meancolumn)
            contentcolumn.height = 500

        add_row = pn.widgets.Button(name="Add aggregation row")
        clear_rows = pn.widgets.Button(name="Clear additional rows")

        # Add interactivity
        clear_rows.on_click(lambda _: remove_column())
        add_row.on_click(lambda _: create_column())

        # this keeps the url in sync with the parameter choices and vice versa

        if pn.state.location:
            other_cntrls = {
                "pick_basin": "pick_basin",
                "pick_GDAC": "pick_GDAC",
                "pick_dsids": "pick_dsids",
                "pick_toggle": "pick_toggle",
                "pick_show_ctrls": "pick_show_ctrls",
                # "pick_variable": "pick_variable", # replaced by pick_variables
                "pick_variables": "pick_variables",
                "pick_scatter_bool": "pick_scatter_bool",
                "pick_scatter_x": "pick_scatter_x",
                "pick_scatter_y": "pick_scatter_y",
                "pick_scatter": "pick_scatter",
                "pick_aggregation": "pick_aggregation",
                "pick_aggregation_method": "pick_aggregation_method",
                "pick_mld": "pick_mld",
                # "pick_mean": "pick_mean",
                "pick_cnorm": "pick_cnorm",
                # "pick_TS": "pick_TS",
                # "pick_profiles": "pick_profiles",
                "pick_activate_scatter_link": "pick_activate_scatter_link",
                "pick_contours": "pick_contours",
                "pick_high_resolution": "pick_high_resolution",
                "pick_startX": "pick_startX",
                "pick_endX": "pick_endX",
                "pick_startY": "pick_startY",
                "pick_endY": "pick_endY",
                "pick_display_threshold": "pick_display_threshold",
                "pick_contour_height": "pick_contour_height",
                "pick_show_decoration": "pick_show_decoration",
                "pick_autorange": "pick_autorange",
                "pick_TS_color_variable": "pick_TS_color_variable",
            }
            pn.state.location.sync(
                self,
                other_cntrls,
            )

        content = self.create_dynmap
        contentcolumn = pn.Column(
            pn.panel(content),  # , defer_load=True),
            # glider_dashboard.create_mean,
            pn.Param(
                self,
                parameters=["pick_show_ctrls"],
                show_name=False,
            ),
            sizing_mode="stretch_width",
            # pn.Row( "# Add data aggregations (mean, max, std...)", button_cols),
        )

        def create_cbar_cntrl(variable):
            return pn.Param(
                self,
                parameters=[f"pick_cbar_range_{variable}"],
                show_name=False,
                widgets={
                    f"pick_cbar_range_{variable}": pn.widgets.EditableRangeSlider(
                        value=(
                            self.stats.loc["1%"][variable],
                            self.stats.loc["99%"][variable],
                            # dictionaries.ranges_dict.get(variable, (0, 10))[0],
                            # dictionaries.ranges_dict.get(variable, (0, 10))[1],
                        ),
                        start=self.stats.loc["1%"][
                            variable
                        ],  # dictionaries.ranges_dict.get(variable, (0, 10))[0],
                        end=self.stats.loc["99%"][
                            variable
                        ],  # dictionaries.ranges_dict.get(variable, (0, 10))[1],
                        # step=0.1,
                    )
                },
            )

        colorbar_widgets_dict = {}
        for variable in lod.variables_selectable:
            colorbar_widgets_dict[f"pick_cbar_range_{variable}"] = (
                pn.widgets.EditableRangeSlider
            )

        controls_accordion = pn.Accordion(
            # toggle=True, # allows only one card to be opened at a time
            objects=[
                (
                    "Choose dataset(s)",
                    pn.Param(
                        self,
                        parameters=["pick_toggle", "pick_basin", "pick_dsids"],
                        widgets={
                            "pick_toggle": {
                                "type": pn.widgets.RadioButtonGroup,
                                "button_type": "success",
                            },
                            "pick_dsids": pn.widgets.MultiChoice,
                        },
                    ),
                ),
                (
                    "Contour plot options",
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
                            "pick_TS_color_variable": pn.widgets.AutocompleteInput,
                        },
                    ),
                ),
                (
                    "more",
                    pn.Param(
                        self,
                        parameters=[
                            "pick_mld",
                            "pick_high_resolution",
                            "pick_show_decoration",
                        ],
                    ),
                ),
                (
                    "Adjust colorbars",
                    pn.Column(
                        pn.Param(
                            self,
                            parameters=["pick_autorange"]
                            + [
                                f"pick_cbar_range_{variable}"
                                for variable in lod.variables_selectable
                            ],
                            widgets=colorbar_widgets_dict,
                            show_name=False,
                        ),
                        "upper and lower boundaries can be overwritten manually.",
                    ),
                ),
            ],
        )

        # print(colorbar_widgets_dict)
        layout = pn.Column(
            pn.Row(
                # row with controls, trajectory plot and TS plot
                controls_accordion,
                pn.Spacer(width=50),
                contentcolumn,
            ),
            pn.Row(pn.Column(), self.markdown),
            pn.Row(),  # Important placeholder for dynamic profile plots, created in glider_dashboard.location
        )
        # print([f"pick_cbar_range_{variable}" for variable in self.pick_variables])
        # it is necessary to hide the controls as a very last option, because hidden controls cannot be accessed as variables
        # in the control flow above. So hiding the controls earlier "defaults" all url and manual settings.
        if self.pick_show_ctrls == False:
            layout[0][0].visible = self.pick_show_ctrls

        try:
            # In case user is updating preexisting page
            self.mylayout.clear()
        except:
            self.mylayout = pn.Row()

        self.mylayout.append(layout)
        return self.mylayout  # mylayout


class MetaDashboard(param.Parameterized):
    options = [
        "glider_serial",
        "optics_serial",
        "altimeter_serial",
        "irradiance_serial",
        "project",
    ]
    options += list(lod.all_metadata.columns)

    pick_serial = param.ObjectSelector(
        default="glider_serial",
        objects=options,
        label="Equipment Ser. No.",
        doc="Track equipment or gliders",
    )

    @param.depends(
        "pick_serial"
    )  # outcommenting this means just depend on all, redraw always
    def create_timeline(self):
        dfm = lod.all_metadata.sort_values("basin")
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

        # Add range slider
        fig.update_layout(
            title=dims,
            xaxis=dict(
                rangeslider=dict(visible=True),
            ),
        )

        for shape in fig["data"]:
            shape["opacity"] = 0.7
        for i, d in enumerate(fig.data):
            d.width = (lod.metadata.deployment_id % 2 + 10) / 12
        fig.layout.autosize = True
        fig.update_layout(height=400)
        return fig


def create_meta_instance(self):
    meta_dashboard = MetaDashboard()
    myrow = pn.Row(
        pn.Column(
            meta_dashboard.param,
            height=500,
        ),
        pn.Column(
            meta_dashboard.create_timeline,
            height=500,
        ),
        height=1500,
        scroll=True,
    )

    try:
        # In case user is updating preexisting page
        mylayout.clear()
    except:
        mylayout = pn.Row()
    mylayout.append(
        pn.widgets.Tabulator(
            lod.all_metadata.reset_index()[
                [
                    "datasetID",
                    "basin",
                    "time_coverage_start (UTC)",
                    "time_coverage_end (UTC)",
                    "date_issued (UTC)",
                    "project",
                    "glider_model",
                    "glider_serial",
                    # "glider_name",
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
            ],
            frozen_columns=["DatasetID"],
            header_filters=True,
            sizing_mode="stretch_both",
            page_size=20,
        )
    )
    return mylayout


GDB = GliderDashboard()

# thecontrols = GDB.create_dynmap()
#
thecontrols = GDB.create_app_instance()


def home():
    pn.panel(pn.Column(thecontrols, "[Open Metadata tables](?page=page1)")).servable(
        title="Glider Dashboard"
    )


def page1():
    pn.panel(
        pn.Column(
            create_meta_instance("self"), " [Return to data visualisation](?page=home)"
        )
    ).servable(title="Metadata tables")


PAGES = {"home": home, "page1": page1}


def get_page_name():
    return pn.state.session_args.get("page", [b"home"])[0].decode(("utf8"))


page_name = get_page_name()
page_func = PAGES[page_name]
page_func()
# pn.serve(APP_ROUTES, title={'app1': 'Some title', 'app2': 'Some other title'}, port=14034)
