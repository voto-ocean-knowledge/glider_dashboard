import time

import cmocean
import datashader as dsh
import holoviews as hv
import hvplot.polars  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param
import plotly.express as px
import polars as pl
import polars.selectors as cs
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

# import initialize
import utils

# IDEA: ON INITIALIZATION, SET OWN minTtime/maxTime (UTC) values based on the
# lazy parquet statistics included in all the files.,

pn.extension(
    "plotly",
    "mathjax",
    "tabulator",
)  # mathjax is currently not used, but could be cool to render latex in markdown
# cudf support works, but is currently not faster
#


# all_metadata is loaded for the metadata visualisation
# all_metadata, allDatasets = utils.load_metadata()
all_metadata = utils.load_metadata_VOTO()

# allDatasets = pd.concat([allDatasetsVOTO, allDatasetsGDAC])

###### filter metadata to prepare download ##############
# metadata, all_datasets = utils.filter_metadata()
metadata = utils.filter_metadata()

metadata = metadata.drop(
    ["nrt_SEA067_M15", "nrt_SEA079_M14", "nrt_SEA061_M63"], errors="ignore"
)  # temporary data inconsistency
metadata["time_coverage_start (UTC)"] = metadata[
    "time_coverage_start (UTC)"
].dt.tz_convert(None)
metadata["time_coverage_end (UTC)"] = metadata["time_coverage_end (UTC)"].dt.tz_convert(
    None
)
allDatasetsVOTO = utils.load_allDatasets_VOTO()
if utils.GDAC_data:
    allDatasetsGDAC = utils.load_allDatasets_GDAC()
    allDatasets = pd.concat([allDatasetsVOTO, allDatasetsGDAC])
else:
    allDatasets = allDatasetsVOTO

dsdict = {}

# HERE I MUST DIFFERENTIATE INTO TWO DATASETS: ONE COMPLETE METADATA FOR THE METADASHBOARD,
# AND ONE/TWO DATASETS THAT ARE FILTERED AND SHWON IN THE DASHBOARD
# all_dataset_names = list(all_datasets.index) + list(metadata.index)
# MUST BE REIMPLEMENTED!!!
# all_dataset_names = set(
#    list(all_datasets.index)
#    + [element.replace("nrt", "delayed") for element in metadata.index]
#    + list(metadata.index)
# )
# )
# CURRENTLY I HAVE AN UNCLEAR DOUBLE FILTER EFFECT HERE :()
all_dataset_names = set(allDatasetsVOTO.index).intersection(
    [element.replace("nrt", "delayed") for element in metadata.index]
)
all_dataset_names = list(all_dataset_names)
all_dataset_names += list(
    metadata.index
)  # Add nrt data because I currently use it for statistics
if utils.GDAC_data:
    all_dataset_names += list(allDatasetsGDAC.index)
#   + list(metadata.index)
# )
all_dataset_names = list(all_dataset_names) + [
    dataset_name + "_small" for dataset_name in all_dataset_names
]

fDs = allDatasets.loc[[name for name in all_dataset_names if "_small" not in name]]
fDs["minTime (UTC)"] = fDs["minTime (UTC)"].dt.tz_localize(None)
fDs["maxTime (UTC)"] = fDs["maxTime (UTC)"].dt.tz_localize(None)
allDatasets["minTime (UTC)"] = allDatasets["minTime (UTC)"].dt.tz_localize(None)
allDatasets["maxTime (UTC)"] = allDatasets["maxTime (UTC)"].dt.tz_localize(None)

# import pdb

# pdb.set_trace()

for dsid in list(allDatasetsVOTO.index) + [
    id + "_small" for id in allDatasetsVOTO.index
]:
    if dsid not in all_dataset_names:
        continue
    dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")

if utils.GDAC_data:
    for dsid in list(allDatasetsGDAC.index) + [
        id + "_small" for id in allDatasetsGDAC.index
    ]:
        dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")
        dsdict[dsid] = (
            dsdict[dsid]
            .drop(cs.string())
            .with_columns(
                pl.col("time").dt.cast_time_unit("ns").dt.replace_time_zone(None)
                # .cast(pl.Float32, strict=False) # if this is activated, time is cast into float32, which leads to bugs in keeping x-range across parameter changes
            )
            .rename({"profile_id": "profile_num"})
        )

# import pdb; pdb.set_trace();
# for dsid in all_dataset_names:
# dsdict[dsid.replace("nrt", "delayed")] = pl.scan_parquet(
#    f"../voto_erddap_data_cache/{dsid.replace('nrt', 'delayed')}.parquet"
# )
"""
print(f"now reading in {dsid}")
if (dsid in list(metadata.index)) or (
    dsid in [element.replace("nrt", "delayed") for element in metadata.index]
):
    dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")
else:
    dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")
    dsdict[dsid] = (
        dsdict[dsid]
        .drop(cs.string())
        .with_columns(
            pl.col("time")
            .dt.cast_time_unit("ns")
            .dt.replace_time_zone(None)
            .cast(pl.Float32, strict=False)
        )
        #.rename({"profile_id": "profile_num"})
    )
    print(dsdict[dsid].collect()['profile_id'])
"""


variables_selectable = (
    pl.concat(dsdict.values(), how="diagonal_relaxed").collect_schema().names()
)

####### specify global plot variables ####################
# df.index = cudf.to_datetime(df.index)
text_opts = hv.opts.Text(text_align="left", text_color="black")
ropts = dict(
    toolbar="above",
    tools=["xwheel_zoom", "reset", "xpan", "ywheel_zoom", "ypan"],
    default_tools=[],
    active_tools=["xpan", "xwheel_zoom"],
    bgcolor="dimgrey",
    # ylim=(-8,None)
)


def plot_limits(plot, element):
    # function to limit user interaction. Can prevent crashes
    # caused by data before 0AD, data out of range...
    plot.handles["x_range"].min_interval = np.timedelta64(2, "h")
    plot.handles["x_range"].max_interval = np.timedelta64(
        int(5 * 3.15e7), "s"
    )  # 5 years
    plot.handles["y_range"].min_interval = 10
    plot.handles["y_range"].max_interval = 500


def mixed_layer_depth(ds, variable, thresh=0.01, ref_depth=-10, verbose=True):
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
    groups = group_by_profiles(ds, [variable, "depth"])
    mld = groups.apply(mld_profile, variable, thresh, ref_depth, verbose)
    return mld


def group_by_profiles(ds, variables=None):
    """
    Group profiles by dives column. Each group member is one dive. The
    returned profiles can be evaluated statistically, e.g. by
    pandas.DataFrame.mean or other aggregating methods. To filter out one
    specific profile, use xarray.Dataset.where instead.

    Parameters
    ----------
    ds : xarray.Dataset
        1-dimensional Glider dataset
    variables : list of strings, optional
        specify variables if only a subset of the dataset should be grouped
        into profiles. Grouping only a subset is considerably faster and more
        memory-effective.
    Return
    ------
    profiles:
    dataset grouped by profiles (dives variable), as created by the
    pandas.groupby methods.
    """
    ds = ds.reset_coords().to_pandas().reset_index().set_index("dives")
    if variables:
        return ds[variables].groupby("dives")
    else:
        return ds.groupby("dives")


def mld_profile(df, variable, thresh, ref_depth, verbose=True):
    exception = False
    divenum = df.index[0]
    df = df.dropna(subset=[variable, "depth"])
    if len(df) == 0:
        mld = np.nan
        exception = True
        message = """no observations found for specified variable in dive {}
                """.format(divenum)
    elif np.nanmin(np.abs(df.depth.values + ref_depth)) > 5:
        exception = True
        message = """no observations within 5 m of ref_depth for dive {}
                """.format(divenum)
        mld = np.nan
    else:
        direction = 1 if np.unique(df.index % 1 == 0) else -1
        # create arrays in order of increasing depth
        var_arr = df[variable].values[:: int(direction)]
        depth = df.depth.values[:: int(direction)]
        # get index closest to ref_depth
        i = np.nanargmin(np.abs(depth + ref_depth))
        # create difference array for threshold variable
        dd = var_arr - var_arr[i]
        # mask out all values that are shallower then ref_depth
        dd[depth > ref_depth] = np.nan
        # get all values in difference array within treshold range
        mixed = dd[abs(dd) > thresh]
        if len(mixed) > 0:
            idx_mld = np.argmax(abs(dd) > thresh)
            mld = depth[idx_mld]
        else:
            exception = True
            mld = np.nan
            message = """threshold criterion never true (all mixed or \
                shallow profile) for profile {}""".format(divenum)
    if verbose and exception:
        print(message)
    return mld


def create_single_ds_plot_raster(data, variables):
    # https://stackoverflow.com/questions/32318751/holoviews-how-to-plot-dataframe-with-time-index
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


def create_cbar_range(variable):
    return param.Range(
        default=(
            0,
            1,
            # dictionaries.ranges_dict[variable][0],
            # dictionaries.ranges_dict[variable][1],
        ),
        # default=(-2, 30), # this is not respected anyway, but below in redefinition
        # step=0.5,
        doc=f"Cbar limits for {variable}",
        precedence=-10,
    )


cbar_range_sliders = {
    f"pick_cbar_range_{variable}": create_cbar_range(variable)
    for variable in variables_selectable
}


class GliderDashboard(param.Parameterized):
    pick_display_threshold = param.Number(
        default=1, step=1, bounds=(-10, 10), label="display_treshold"
    )
    pick_variables = param.ListSelector(
        default=["temperature"],
        allow_None=False,
        objects=variables_selectable,
        label="variable",
        doc="Variable used to create colormesh",
        precedence=1,
    )

    pick_scatter_x = param.Selector(
        default=None,  # "salinity",
        allow_None=False,
        objects=variables_selectable,
        label="X-axis variable",
        doc="Variable used to create colormesh",
        precedence=-10,
    )

    pick_scatter_y = param.Selector(
        default=None,  # "temperature",
        allow_None=False,
        objects=variables_selectable,
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
    alldslist = list(filter(lambda k: "nrt" in k, dsdict.keys()))
    alldslist = [x for x in alldslist if "_small" not in x]
    if utils.GDAC_data:
        alldslist += list(allDatasetsGDAC.index)
    alldslabels = [
        element[4:] if element[0:4] == "nrt_" else element for element in alldslist
    ]
    objectsdict = dict(zip(alldslabels, alldslist))

    pick_dsids = param.ListSelector(
        default=[],  # [alldslist[0]],#dslist[0]],
        objects=objectsdict,  # alldslist,
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

    (locals().update(cbar_range_sliders),)  # noqa

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
        default=allDatasets["minTime (UTC)"].min(),
        label="startX",
        doc="startX",
        precedence=1,
    )
    pick_endX = param.Date(
        default=allDatasets["maxTime (UTC)"].max(),
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
        precedence=1,
    )
    # pick_TS = param.Boolean(
    #     default=False,
    #     label="Show TS-diagram",
    #     doc="Activate salinity-temperature diagram",
    #     precedence=1,
    # )
    pick_TS_color_variable = param.Selector(
        default=None,
        objects=variables_selectable,
        label="Colour scatterplot by",
        doc="blubb",
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
            + variables_selectable
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
    # file_download = None
    contour_processing = False
    startX, endX = (
        # metadata["time_coverage_start (UTC)"].min().to_datetime64(),
        fDs["minTime (UTC)"].min().to_datetime64(),
        # - np.timedelta64(6 * 30 * 24, "s"),  # last six months
        fDs["maxTime (UTC)"].max().to_datetime64(),
    )

    annotations = []

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
        p3 = f"""from {np.datetime_as_string(self.startX, unit="s")} to {np.datetime_as_string(self.endX, unit="s")}. """

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

    @param.depends("pick_display_threshold", watch=True)
    def update_display_threshold(self):
        for var in [
            "pick_variables",
            "pick_basin",
            "pick_toggle",
            "pick_dsids",
            "pick_cnorm",
            "pick_aggregation",
            "pick_mld",
            # se
            # "pick_mean",
            # "pick_TS",
            # "pick_profiles",
            "pick_activate_scatter_link",
            "pick_contours",
            "pick_high_resolution",
            "button_inflow",
        ]:
            self.param[var].precedence = self.pick_display_threshold

    @param.depends("pick_show_ctrls", watch=True)
    def update_display_threshold(self):
        try:
            # first run, when layout does not exist, this fails deliberately.
            mylayout[0][0][0].visible = self.pick_show_ctrls
            # print(mylayout)
        except:
            pass

    @param.depends("pick_toggle", "pick_basin", watch=True)
    def update_datasource(self):
        # toggles visibility
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
        # for i in range(10,20):
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
            meta = metadata[metadata["basin"] == self.pick_basin]
            meta = meta[meta["project"] == "SAMBA"]
            meta = meta[meta["time_coverage_start (UTC)"] > np.datetime64("2021-01-01")]
            meta = utils.drop_overlaps_fast(meta)
            meta = fDs.loc[meta.index]
        else:
            # second case, user selected dids
            meta = allDatasets.loc[self.pick_dsids]  # metadata.loc[self.pick_dsids]

        # hacky way to differentiate if called via synclink or refreshed with UI buttons
        if not len(meta):
            self.startX = None
            self.endX = None
            self.pick_startX = None
            self.pick_endX = None
            return
        incoming_link = not (isinstance(self.pick_startX, pd.Timestamp))
        # print('ISINSTANCE', isinstance(self.pick_startX, pd.Timestamp))
        # print('INCOMING VIA LINK:', incoming_link)
        # import pdb

        # pdb.set_trace()
        if not incoming_link:
            mintime = meta["minTime (UTC)"].min()
            maxtime = meta["maxTime (UTC)"].max()
            self.startX, self.endX = mintime, maxtime
            self.pick_startX, self.pick_endX = (mintime, maxtime)
        else:
            self.pick_startX, self.pick_endX = (self.pick_startX, self.pick_endX)

        self.startY = None
        self.endY = 12

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
            profilelabel = (
                "descending"
                if profile.select(pl.col("profile_direction").mean())[0, 0] > 0
                else "ascending"
            )
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
            # print("ITEMS:", items)
            profile_plots.append(
                hv.Overlay(items=items).opts(
                    legend_position="bottom_right", show_legend=True
                )
            )
        mylayout[0][2] = pn.Row(hv.Layout(profile_plots))

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
        mylayout.append(self.file_download)

    @param.depends(
        "pick_dsids",
        "pick_toggle",
        "pick_basin",
        watch=True,
    )
    def update_data(self):
        x_range = (self.startX, self.endX)
        meta, plt_props = self.load_viewport_datasets(x_range)

        # if plt_props["zoomed_out"]:
        #    metakeys = [element.replace("nrt", "delayed") for element in meta.index]
        # else:
        metakeys = [
            (
                element.replace("nrt", "delayed")
                if element.replace("nrt", "delayed") in allDatasets.index
                else element
            )
            for element in meta.index
        ]
        varlist = []
        for dsid in metakeys:
            # This is delayed data if available
            if plt_props["zoomed_out"]:
                ds = dsdict[dsid + "_small"]
            else:
                ds = dsdict[dsid]

            # ds = ds.filter(pl.col("profile_num") % plt_props["subsample_freq"] == 0)
            varlist.append(ds)

        # This should only be a temporay hack. I don't want all that data to go into my TS plots.
        # dsconc = utils.voto_concat_datasets2(varlist)
        dsconc = pl.concat([data for data in varlist], how="diagonal_relaxed")
        dsconc = dsconc.with_columns(pl.col("depth").neg()).sort("time")
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
        *list(cbar_range_sliders.keys()),  # noqa
        "pick_autorange",
        "pick_TS_color_variable",
        # watch=True,
    )  # outcommenting this means just depend on all, redraw always
    def create_dynmap(self):
        # self.markdown.object = self.update_markdown()

        self.startX = self.pick_startX
        self.endX = self.pick_endX

        self.startY, self.endY = (self.pick_startY, self.pick_endY)

        # self.startY = self.pick_startY
        # self.endY = self.pick_endY

        # in case coming in over json link
        self.startX = np.datetime64(self.startX)
        self.endX = np.datetime64(self.endX)

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
                self.pick_scatter_x = "temperature"  # self.pick_variables[0]
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
        x_range = (self.startX, self.endX)
        y_range = (self.startY, self.endY)

        range_stream = RangeXY(x_range=x_range, y_range=y_range).rename()
        range_stream.add_subscriber(self.keep_zoom)
        # range_stream.add_subscriber(self.update_markdown) # Is always one step after, thus deactivated here

        # Create a callback for a dynamic map
        tap_stream = Tap(x=np.nan, y=np.nan)
        tap_stream.add_subscriber(self.location)

        pick_cnorm = "linear"

        dmap_raster = hv.DynamicMap(
            self.get_xsection_raster,
            streams=[range_stream, tap_stream],
        )
        # t1 = time.perf_counter()
        # if (self.data_in_view is not None) and

        # t2 = time.perf_counter()
        # print("statistical operations took:", t2 - t1)

        if self.pick_high_resolution:
            pixel_ratio = 1.0
        else:
            pixel_ratio = 0.5
        # if self.pick_aggregation=='var':
        #    means = dsh.var(self.pick_variable)

        # initialize dictionary for resulting plots:
        # plot_elements_dict = dict()

        if self.pick_scatter_bool:
            dmap_TS = hv.DynamicMap(
                self.get_xsection_TS,
                streams=[range_stream],
                cache_size=1,
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

            # dmapTSr = rasterize(
            #    dmap_TS,
            #    pixel_ratio=pixel_ratio,
            # ).opts(
            #    cnorm="eq_hist",
            # )

        """
            dcont = hv.DynamicMap(
                self.get_density_contours, streams=[range_stream]
            ).opts(
                alpha=0.5,
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
                    cnorm="eq_hist",
                    cmap=dictionaries.cmap_dict.get(
                        self.pick_TS_color_variable, cmocean.cm.solar
                    ),
                    # clabel=f"{self.pick_variable}  [{dictionaries.units_dict[self.pick_variable]}]",
                    colorbar=True,
                )

        if self.pick_profiles:
            dmap_profiles = hv.DynamicMap(
                self.get_xsection_profiles,
                streams=[range_stream],
                cache_size=1,
            )
            dmap_profilesr = rasterize(
                dmap_profiles,
                pixel_ratio=pixel_ratio,
            ).opts(
                cnorm="eq_hist",
            )
        """

        dmap_decorators = hv.DynamicMap(
            self.get_xsection, streams=[range_stream], cache_size=1
        )
        if self.pick_mld:
            # Important!!! Compute MLD only once and apply it to all plots!!!
            dmap_mld = hv.DynamicMap(
                self.get_xsection_mld, streams=[range_stream], cache_size=1
            )  # .opts(responsive=True)

        # cntr_plts = []
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
                # invert_yaxis=True, # Would like to activate this, but breaks the hover tool
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
                clabel=f"{variable}  [{dictionaries.units_dict.get(variable, '')}]",  # self.pick_variable,
                clim_percentile=True if self.pick_autorange else False,
                fontscale=2,
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
                    ylim=(self.startY, self.endY),
                    xlim=(self.startX, self.endX),
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
                    ylim=(self.startY, self.endY),
                    xlim=(self.startX, self.endX),
                    hooks=[lambda p, _: p.state.update(border_fill_alpha=0)],
                )
            if self.pick_show_decoration:
                plots_dict["dmap_rasterized"][variable] = plots_dict["dmap_rasterized"][
                    variable
                ].opts(ylim=(None, 12))
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
                    [element for element in plots_dict["dmap_rasterized"].values()]
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
                diffx = (
                    self.stats.loc["99%"][self.pick_scatter_x]
                    - self.stats.loc["5%"][self.pick_scatter_x]
                )
                xlim = (
                    self.stats.loc["5%"][self.pick_scatter_x] - 0.1 * diffx,
                    self.stats.loc["99%"][self.pick_scatter_x] + 0.1 * diffx,
                )
                diffy = (
                    self.stats.loc["99%"][self.pick_scatter_y]
                    - self.stats.loc["5%"][self.pick_scatter_y]
                )

                ylim = (
                    self.stats.loc["1%"][self.pick_scatter_y] - 0.1 * diffy,
                    self.stats.loc["99%"][self.pick_scatter_y] + 0.1 * diffy,
                )
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
            contourplots = contourplots * dmap_decorators
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
        return pn.Column(contourplots.cols(ncols))

    def create_mean(self):
        self.startX = self.pick_startX
        self.endX = self.pick_endX

        # in case coming in over json link
        self.startX = np.datetime64(self.startX)
        self.endX = np.datetime64(self.endX)
        x_range = (self.startX, self.endX)
        y_range = (self.startY, self.endY)
        range_stream = RangeXY(x_range=x_range, y_range=y_range).rename()
        # dmap_raster = hv.DynamicMap(
        #    self.get_xsection_raster,
        #    streams=[range_stream],
        # )
        dmap = hv.DynamicMap(self.get_xsection, streams=[range_stream], cache_size=1)
        dmap_mean = (
            hv.DynamicMap(
                self.get_xsection_mean, streams=[range_stream], cache_size=1
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
        dt = x1 - x0
        dtns = dt / np.timedelta64(1, "ns")
        plt_props = {}
        try:
            # necessary if changing dsids dynamically
            x0 = x0.to_datetime64()
            x1 = x1.to_datetime64()
        except:
            pass
        # filtered Datasets

        # pdb.set_trace()
        # fDs = allDatasets.loc[
        #    [name for name in all_dataset_names if "_small" not in name]
        # ]
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

        # print(fD_inview)
        # mydslist = [name for name in all_dataset_names if '_small' not in name]

        if self.pick_toggle == "SAMBA obs.":
            # first case, , user selected an aggregation, e.g. 'Bornholm Basin'
            #
            fD_inview = fD_inview[
                fD_inview["institution"] == "Voice of the Ocean Foundation"
            ]
            meta = metadata.loc[
                [name for name in fD_inview.index if "delayed" not in name]
            ]
            meta = meta[meta["basin"] == self.pick_basin]
            meta = meta[meta["project"] == "SAMBA"]
            meta = utils.drop_overlaps_fast(meta)

        else:
            # second case, user selected dids
            # try:
            #     meta = metadata.loc[self.pick_dsids]
            # except:
            meta = allDatasets.loc[self.pick_dsids]
            # meta["time_coverage_start (UTC)"] = meta["minTime (UTC)"]
            # meta["time_coverage_end (UTC)"] = meta["maxTime (UTC)"]

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
        return allDatasets.loc[meta.index], plt_props

    def get_xsection_mld(self, x_range, y_range):
        try:
            dscopy = utils.add_dive_column(self.data_in_view).compute()
        except:
            dscopy = utils.add_dive_column(self.data_in_view)
        # dscopy["depth"] = -dscopy["depth"]
        mld = mixed_layer_depth(
            dscopy.to_xarray(), "temperature", thresh=0.3, verbose=False, ref_depth=5
        )
        gtime = dscopy.reset_index().groupby(by="profile_num").mean().time
        dfmld = (
            pd.DataFrame.from_dict(
                dict(time=gtime.values, mld=mld.rolling(10, center=True).mean().values)
            )
            .sort_values(by="time")
            .dropna()
        )

        mldscatter = dfmld.hvplot.line(
            x="time",
            y="mld",
            color="white",
            alpha=0.5,
            responsive=True,
        )
        return mldscatter

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

    def get_xsection_raster(self, x_range, y_range, x, y):  # , contour_variable=None):
        (x0, x1) = x_range
        self.pick_startX = pd.to_datetime(x0)  # setters
        self.pick_endX = pd.to_datetime(x1)
        meta, plt_props = self.load_viewport_datasets(x_range)

        # if plt_props["zoomed_out"]:
        #    metakeys = [element.replace("nrt", "delayed") for element in meta.index]
        # else:
        # import pdb

        # pdb.set_trace()
        metakeys = [
            (
                element.replace("nrt", "delayed")
                if element.replace("nrt", "delayed") in allDatasetsVOTO.index
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
            if self.pick_scatter_x:
                variables = variables + [self.pick_scatter_x]
            if self.pick_scatter_y:
                variables = variables + [self.pick_scatter_y]
            if self.pick_TS_color_variable:
                variables = variables + [self.pick_TS_color_variable]
                # self.pick_scatter_y,
                # self.pick_TS_color_variable,
            # ]
        variables = list(set(variables))
        varlist = []
        varlist_small = []
        # if plt_props["zoomed_out"]:
        # import pdb; pdb.set_trace();
        for dsid in metakeys:
            # This is delayed data if available
            if plt_props["zoomed_out"] and (not self.pick_high_resolution):
                ds = dsdict[dsid + "_small"]
            else:
                ds = dsdict[dsid]  # + "_small"]

            # ds = ds.filter(pl.col("profile_num") % plt_props["subsample_freq"] == 0)
            varlist.append(ds)

        for dsid in meta.index:
            # This is only the nrt data
            ds = dsdict[dsid]
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
        dsconc = dsconc.with_columns(pl.col("depth").neg()).sort("time")

        dsconc_small = utils.voto_concat_datasets2(varlist_small)
        dsconc_small = dsconc_small.with_columns(pl.col("depth").neg()).sort("time")

        self.data_in_view = dsconc  # .dropna(subset=['temperature', 'salinity'])
        self.data_in_view_small = dsconc_small

        # print(
        #    f"the length of dsconc is now {dsconc.collect().height}\n and the length of dsconc_small is {dsconc_small.collect().height}"
        # )

        # THIS IS EXPENSIVE. I SHOULD CREATE STATS ONLY WHERE NEEDED; ESPECIALLY WITH .to_pandas()
        self.stats = (
            self.data_in_view_small.select(variables)
            .describe(  # .select(pl.col(self.pick_variables))
                (0.01, 0.05, 0.99)
            )
            .to_pandas()
            .set_index("statistic")
        )

        # THIS MUST BE REMOVE FOR GREAT PERFORMANCE.
        # REQUIRES REWRITE OF SOME CLIM AND QUANTILE FILTERS I BELIEVE
        # self.update_markdown(x_range, y_range)  # THIS SHOULD BE READDED EVENTUALLY

        mplt = create_single_ds_plot_raster(data=self.data_in_view, variables=variables)
        return mplt

    def get_xsection_TS(self, x_range, y_range):
        # dsconc = self.data_in_view.filter(pl.col("salinity") > 1)
        # t1 = time.perf_counter()
        # stats = dsconc.select(pl.col("temperature", "salinity")).describe((0.01, 0.99))

        # low = #stats.filter(pl.col("statistic") == "1%")
        # high = #stats.filter(pl.col("statistic") == "99%")

        # t2 = time.perf_counter()
        # if self.pick_variables[0]
        # Needs additional variable.
        vdims = ["depth", "time"]
        if self.pick_TS_color_variable:
            vdims.append(self.pick_TS_color_variable)
        mplt = hv.Points(
            data=self.data_in_view,
            kdims=[self.pick_scatter_x, self.pick_scatter_y],
            vdims=vdims,  # self.pick_TS_color_variable if self.pick_TS_color_variable else None,
            # list(variables),
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
                # * hv.Text(
                #    x=self.startX,
                #    y=-20,
                #    text="There is no data here!",
                #    fontsize=10,)
            )
        elif type == "Spikes":
            element = hv.Spikes().opts(color="black", alpha=0.1)
        return element

    def get_xsection(self, x_range, y_range):
        (x0, x1) = x_range
        try:
            # necessary if changing dsids dynamically
            x0 = x0.to_datetime64()
            x1 = x1.to_datetime64()
        except:
            pass
        t1 = time.perf_counter()
        meta, plt_props = self.load_viewport_datasets(x_range)

        meta_start_in_view = meta[(meta["minTime (UTC)"] > x0)]
        meta_end_in_view = meta[(meta["maxTime (UTC)"] < x1)]

        startvlines = (
            hv.VLines(meta_start_in_view["minTime (UTC)"]).opts(
                color="grey", line_width=1
            )  # , spike_length=20)
            # .opts(position=-10)
        )
        endvlines = (
            hv.VLines(meta_end_in_view["maxTime (UTC)"]).opts(
                color="grey", line_width=1
            )  # , spike_length=20)
            # .opts(position=-10)
        )
        """
        startvlines = (
            hv.Vlines(meta_start_in_view["time_coverage_start (UTC)"])
            .opts(color="grey")
            #.opts(position=-10)
        )
        endvlines = (
            hv.Vlines(meta_end_in_view["time_coverage_end (UTC)"])
            .opts(color="red")
            #.opts(position=-10)
        )
        """

        data = pd.DataFrame.from_dict(
            dict(
                time=meta_start_in_view["minTime (UTC)"].values,
                y=5,
                text=meta_start_in_view.index.str.replace("nrt_", ""),
            )
        )
        ds_labels = hv.Labels(data).opts(
            fontsize=12,
            text_align="left",  # plt_props['dynfontsize'],
        )
        plotslist = []
        if len(meta_start_in_view) > 0:
            plotslist.append(startvlines)
            plotslist.append(ds_labels)
        if len(meta_end_in_view) > 0:
            plotslist.append(endvlines)
        if plotslist:
            return hv.Overlay(plotslist)  # reduce(lambda x, y: x*y, plotslist)
        else:
            return hv.Overlay()  # return self.create_None_element("Overlay")


class MetaDashboard(param.Parameterized):
    options = [
        "glider_serial",
        "optics_serial",
        "altimeter_serial",
        "irradiance_serial",
        "project",
    ]
    options += list(all_metadata.columns)

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
        dfm = all_metadata.sort_values("basin")
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
            d.width = (metadata.deployment_id % 2 + 10) / 12
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
        height=500,
        scroll=True,
    )

    mylayout.clear()  # =
    mylayout.append(myrow)
    mylayout.append(
        pn.widgets.Tabulator(
            all_metadata[
                [
                    # "datasetID",
                    "basin",
                    "time_coverage_start (UTC)",
                    "time_coverage_end (UTC)",
                    "date_issued (UTC)",
                    "project",
                    "glider_model",
                    "glider_serial",
                    "variables",
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
                ]
            ],
            header_filters=True,
            layout="fit_data_table",
            # widths=120,
        )
    )
    mylayout.append(pn.Column(button_dash, button_meta))
    return mylayout


@param.depends(
    "pick_show_ctrls",
    watch=True,
)
def create_app_instance(self):
    glider_dashboard = GliderDashboard()

    def create_cbar_cntrl(variable):
        return pn.Param(
            glider_dashboard,
            parameters=[f"pick_cbar_range_{variable}"],
            show_name=False,
            widgets={
                f"pick_cbar_range_{variable}": pn.widgets.EditableRangeSlider(
                    value=(
                        dictionaries.ranges_dict.get(variable, (0, 10))[0],
                        dictionaries.ranges_dict.get(variable, (0, 10))[1],
                    ),
                    start=dictionaries.ranges_dict.get(variable, (0, 10))[0],
                    end=dictionaries.ranges_dict.get(variable, (0, 10))[1],
                    step=0.1,
                )
            },
        )

    cbar_cntrls = [create_cbar_cntrl(variable) for variable in variables_selectable]

    # Data options
    ctrl_data = pn.Column(  # top stack, dataset and basin options
        "Choose input data either based on basin location or ID",
        pn.Param(
            glider_dashboard,
            parameters=["pick_toggle"],
            widgets={
                "pick_toggle": pn.widgets.RadioButtonGroup,
                "button_type": "success",
            },
            # css_classes=["widget-button"],
            # default_layout=pn.Column,
            show_name=False,
            # width=100,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_basin"],
            # widgets={'pick_basin':pn.widgets.MultiChoice(max_items=1)}
            default_layout=pn.Column,
            # max_items=1,
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_dsids"],
            widgets={"pick_dsids": pn.widgets.MultiChoice},
            show_name=False,
        ),
        # styles={"background": "#C0C0C0"},
    )

    # contour plot options
    ctrl_contour = pn.Column(
        pn.Param(
            glider_dashboard,
            parameters=["pick_variables"],
            widgets={
                "pick_variables": pn.widgets.MultiChoice
            },  # pn.widgets.CheckBoxGroup},
            default_layout=pn.Column,
            show_name=False,
        ),
        # pn.widgets.
        pn.Param(
            glider_dashboard,
            parameters=["pick_cnorm"],
            widgets={"pick_cnorm": pn.widgets.RadioButtonGroup},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_aggregation"],
            widgets={"pick_aggregation": pn.widgets.RadioButtonGroup},
            show_name=False,
            show_labels=True,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_contours"],
            show_name=False,
        ),
    )

    # scatter options
    ctrl_scatter = pn.Column(
        # pn.Param(
        #     glider_dashboard,
        #     parameters=["pick_TS", "pick_activate_scatter_link"],
        #     default_layout=pn.Row,
        #     show_name=False,
        #     # display_threshold=10,
        # ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_scatter_bool"],
            widgets={"pick_scatter_bool": pn.widgets.Switch},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_scatter"],
            widgets={"pick_scatter": pn.widgets.RadioButtonGroup},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_scatter_x"],
            widgets={"pick_scatter_x": pn.widgets.AutocompleteInput},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_scatter_y"],
            widgets={"pick_scatter_y": pn.widgets.AutocompleteInput},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_TS_color_variable"],
            widgets={"pick_TS_color_variable": pn.widgets.AutocompleteInput},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_activate_scatter_link"],
            show_name=False,
            # display_threshold=10,
        ),
        # pn.Param(
        #     glider_dashboard,
        #     parameters=["pick_profiles"],
        #     show_name=False,
        #     # display_threshold=10,
        # ),
        # This is a hidden parameter, which can be specified in url
        # to show or hide the menus. Can be useful when emedding interactive
        # figures in webpages or presentations for example.
        # pn.Param(glider_explorer,
        #    parameters=['pick_display_threshold'],
        #    show_name=False,
        #    display_threshold=10,),
    )

    ctrl_more = pn.Column(
        pn.Param(
            glider_dashboard,
            parameters=["startX"],
            show_name=False,
            # display_threshold=10,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_high_resolution"],
            show_name=False,
            # display_threshold=10,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_mld"],
            show_name=False,
            # display_threshold=0.5,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_show_decoration"],
            show_name=False,
            # display_threshold=0.5,
        ),
        # pn.Param(
        #    glider_dashboard,
        #    parameters=["pick_mean"],
        #    show_name=False,
        #    # display_threshold=0.5,
        # ),
        # pn.Param(
        #    glider_dashboard,
        #    parameters=["button_inflow"],
        #    show_name=False,
        #    # display_threshold=10,
        # ),
        # button_cols,
        pn.Param(
            glider_dashboard,
            parameters=["button"],
            show_name=False,
            # widgets={"button": pn.widgets.Button},
            # display_threshold=0.5,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["endX"],
            show_name=False,
            # display_threshold=10,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["startY"],
            show_name=False,
            # display_threshold=10,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["endY"],
            show_name=False,
            # display_threshold=10,
        ),
    )
    ctrl_colorbars = pn.Column(
        pn.Param(
            glider_dashboard,
            parameters=["pick_autorange"],
            show_name=False,
        ),
        *cbar_cntrls,
    )

    pick_aggregation_method = pn.Param(
        glider_dashboard,
        parameters=["pick_aggregation_method"],
        widgets={"pick_aggregation_method": pn.widgets.RadioButtonGroup},
        show_name=False,
        show_labels=True,
    )

    def create_column(hex_id=None):
        """
        Dynamically add a new row to the app.
        """
        # value = random.randint(0, 100)
        # column = pn.widgets.TextInput(name="Enter a number", value=str(value))
        global meancolumn
        meancolumn = pn.Column(
            glider_dashboard.create_mean(),
            height=500,
        )
        contentcolumn.append(meancolumn)
        contentcolumn.height = 1050

    def remove_column(hex_id=None):
        """
        Dynamically remove a column from the app.
        """
        # value = random.randint(0, 100)
        # column = pn.widgets.TextInput(name="Enter a number", value=str(value))
        contentcolumn.remove(meancolumn)
        contentcolumn.height = 500

    add_row = pn.widgets.Button(name="Add aggregation row")
    clear_rows = pn.widgets.Button(name="Clear additional rows")

    # main = pn.Column("# Dynamically add new rows", button_cols, layout)

    # Add interactivity
    clear_rows.on_click(lambda _: remove_column())
    add_row.on_click(lambda _: create_column())

    # this keeps the url in sync with the parameter choices and vice versa
    if pn.state.location:
        cbar_dict = {
            f"pick_cbar_range_{variable}": f"pick_cbar_range_{variable}"
            for variable in variables_selectable
        }
        other_cntrls = {
            "pick_basin": "pick_basin",
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
        other_cntrls.update(cbar_dict)
        pn.state.location.sync(
            glider_dashboard,
            other_cntrls,
        )

    contentcolumn = pn.Column(
        # pn.Row(
        glider_dashboard.create_dynmap,
        # glider_dashboard.create_mean,
        pn.Param(
            glider_dashboard,
            parameters=["pick_show_ctrls"],
            show_name=False,
        ),
        # height=glider_dashboard.pick_contour_heigth,
        # ),
        # pn.Row("# Add data aggregations (mean, max, std...)", button_cols),
    )

    layout = pn.Column(
        pn.Row(  # row with controls, trajectory plot and TS plot
            pn.Accordion(
                # toggle=True, # allows only one card to be opened at a time
                objects=[
                    ("Choose dataset(s)", ctrl_data),
                    ("Contour plot options", ctrl_contour),
                    ("Linked (scatter-)plots", ctrl_scatter),
                    # ('Aggregations (WIP)', pn.Column(
                    #    pick_aggregation_method,
                    #    add_row,
                    #    clear_rows,
                    #    )),
                    ("more", ctrl_more),
                    ("adjust Colorbars", ctrl_colorbars),
                    # ('WIP',add_row),
                ],
            ),
            pn.Spacer(width=50),
            contentcolumn,
            # height=800,
        ),
        pn.Row(pn.Column(), glider_dashboard.markdown),
        pn.Row(),  # Important placeholder for dynamic profile plots, created in glider_dashboard.location
    )

    # it is necessary to hide the controls as a very last option, because hidden controls cannot be accessed as variables
    # in the control flow above. So hiding the controls earlier "defaults" all url and manual settings.
    if glider_dashboard.pick_show_ctrls == False:
        layout[0][0].visible = glider_dashboard.pick_show_ctrls
    mylayout.clear()
    mylayout.append(layout)
    mylayout.append(pn.Column(button_dash, button_meta))
    return mylayout


button_dash = pn.widgets.Button(name="activate Glider Datasets Dashboard")
button_dash.on_click(create_app_instance)

button_meta = pn.widgets.Button(
    name="activate schematic Mission Overview and Metadata Table "
)
button_meta.on_click(create_meta_instance)

# usefull to create secondary plot, but not fully indepentently working yet:
# glider_explorer2=GliderExplorer()
mylayout = pn.Column(button_dash, button_meta)
button_dash.clicks += (
    1  # to activate the Glider Data dashboard from the start as default
)
# mylayout.append(create_app_instance("placeholder"))
mylayout.servable()
# app = layout # create_app_instance()
# app2 = create_app_instance()
# layout.servable()
# app2.servable()
#    port=12345,
#    websocket_origin='*',
#    title='VOTO SAMBA data',
#    threaded=True)

"""
pn.serve(
    create_app_instance,
    port=12345,
    websocket_origin='*',
    title='VOTO SAMBA data',
    threaded=True)


#.show(
#    title='VOTO SAMBA data',
#    websocket_origin='*',
#    port=12345,
    #admin=True,
    #profiler=True
#    )



Future development ideas:
* activate hover (for example dataset details, sensor specs, or point details)
* holoviews autoupdate for development
* write tests including timings benchmark for development
* implement async functionen documented in holoviews to not disturb user interaction
* throw out X_range_stream (possibly) and implement full data dynamic sampling instead. One solution could be to use a dynamic .sample(frac=zoomstufe)
* plot glidertools gridded data instead (optional, but good for interpolation)...
* good example to follow is the AdvancedStockExplorer class in the documentation
* add secondary plot option 'profile', or color in different variables (e.g. the plot variable)
* disentangle interactivity, so that partial refreshes (e.g. mixed layer calculation only) don't trigger complete refresh
* otpimal colorbar range (percentiles?)
* on selection of a new basin, I should reset the ranges. Otherwise it could come up with an error when changing while having unavailable x_range.
* range tool link, such as the metadata plot
* optimize performance with dask after this video: https://www.youtube.com/watch?v=LKIRAzsqLb0
* currently, the import statements make up a substantial part of the initial loading time -> check for unused imports, check enable chaching, check if new import necessary for each intial load. (time saving 4s)
* Should refactor out the three (?) different methods of concatenating datasets and instead have one function to do that all. Then also switch between dataframes/dask should be easier.
* in the xr.open_mfdataset, path all the dataset_ids in a list instead and keep parallel=True activated to drastically improve startup time.Maybe this could even enable live loading of the data, instead of preloading it into the RAM/dask

expensive: datashader_apply spreading 2.464
...
"""
