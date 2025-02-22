import time
import glidertools as gt
import hvplot.dask
import hvplot.pandas
import cmocean
import holoviews as hv
from holoviews import opts
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import rasterize, spread, dynspread, regrid
from holoviews.selection import link_selections

# from bokeh.models import DatetimeTickFormatter, HoverTool
from holoviews.streams import RangeX, RangeXY
import numpy as np
from functools import reduce
import panel as pn
import param
import plotly.express as px
import initialize
import dask
import dask.dataframe as dd
from download_glider_data import utils as dutils
import utils
import dictionaries

pn.extension("plotly")
try:
    # cudf support works, but is currently not faster
    import hvplot.cudf
except:
    print("no cudf available, that is fine but slower")

# all_metadata is loaded for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(
    ["nrt_SEA067_M15", "nrt_SEA079_M14", "nrt_SEA061_M63"], errors="ignore"
)  # temporary data inconsistency
metadata["time_coverage_start (UTC)"] = metadata[
    "time_coverage_start (UTC)"
].dt.tz_convert(None)
metadata["time_coverage_end (UTC)"] = metadata["time_coverage_end (UTC)"].dt.tz_convert(
    None
)
dsdict = initialize.dsdict

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


def create_single_ds_plot_raster(data, variable):
    # https://stackoverflow.com/questions/32318751/holoviews-how-to-plot-dataframe-with-time-index
    raster = data.hvplot.points(
        x="time",
        y="depth",
        c=variable,
    )
    return raster


class GliderDashboard(param.Parameterized):

    pick_variable = param.Selector(
        default="temperature",
        objects=[
            "temperature",
            "salinity",
            "potential_density",
            "chlorophyll",
            "oxygen_concentration",
            "cdom",
            "backscatter_scaled",
            "phycocyanin",
            "phycocyanin_tridente",
            "methane_concentration",
            "longitude",
            "latitude",
        ],
        label="variable",
        doc="Variable used to create colormesh",
        precedence=1,
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
    alldslist = list(filter(lambda k: 'nrt' in k, dsdict.keys()))
    alldslabels = [element[4:] for element in alldslist]
    objectsdict = dict(zip(alldslabels,alldslist))
    # import pdb; pdb.set_trace();
    pick_dsids = param.ListSelector(
        default=[],#[alldslist[0]],#dslist[0]],
        objects=objectsdict,#alldslist,
        label="DatasetID",
        precedence=-10,
    )

    pick_toggle = param.Selector(
        objects=['SAMBA obs.', 'DatasetID'],
        label="choose by SAMBA observatory or data ID",
    )

    pick_cnorm = param.Selector(
        default="linear",
        objects=["linear", "eq_hist", "log"],
        doc="Colorbar Transformations",
        label="Colourbar Scale",
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
        objects=["mean", "min", "max"],#, "std"],
        label="1D Data Aggregation",
        doc="Method that is applied to aggregate column",
        precedence=1,
    )

    pick_mld = param.Boolean(
        default=False, label="MLD", doc="Show Mixed Layer Depth", precedence=1
    )
    #pick_mean = param.Boolean(
    #    default=False, label="mean", doc="Show column mean", precedence=1
    #)
    pick_startX = param.Date(
        default=metadata["time_coverage_start (UTC)"].min(),
        label="startX", doc="startX", precedence=1
    )
    pick_endX = param.Date(
        default=metadata["time_coverage_end (UTC)"].max(),
        label="endX", doc="endX", precedence=1
    )
    pick_startY = param.Number(
        default=None,
        label="startY", doc="startY", precedence=1
    )
    pick_endY = param.Number(
        default=8,
        label="endY", doc="endY", precedence=1
    )
    pick_contour_heigth = param.Number(
        default=550,
        label="contour_heigth", precedence=1
    )
    pick_display_threshold = param.Number(
        default=1, step=1, bounds=(-10, 10), label="display_treshold"
    )
    pick_TS = param.Boolean(
        default=False,
        label="Show TS-diagram",
        doc="Activate salinity-temperature diagram",
        precedence=1,
    )
    pick_profiles = param.Boolean(
        default=False,
        label="Show profiles",
        doc="Activate profiles diagram",
        precedence=1,
    )
    pick_TS_colored_by_variable = param.Boolean(
        default=False,
        label="Colour TS by variable",
        doc='Colours the TS diagram by "variable" instead of "count of datapoints"',
        precedence=1,
    )
    pick_contours = param.Selector(
        default=None,
        objects=[
            None,
            "temperature",
            "salinity",
            "potential_density",
            "chlorophyll",
            "oxygen_concentration",
            "cdom",
            "backscatter_scaled",
            "phycocyanin",
            "phycocyanin_tridente",
            "methane_concentration",
            "longitude",
            "latitude",
        ],
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
    data_in_view = None
    contour_processing = False
    startX, endX = (
        #metadata["time_coverage_start (UTC)"].min().to_datetime64(),
        metadata["time_coverage_end (UTC)"].max().to_datetime64()-np.timedelta64(6*30*24,'s'), # last six months
        metadata["time_coverage_end (UTC)"].max().to_datetime64(),
    )

    #startY, endY = (None, 8)
    #import pdb; pdb.set_trace();
    #
    annotations = []

    def update_markdown(self, x_range, y_range):
        p1 = f"""\
            # About
            Ocean {self.pick_variable} in [{dictionaries.units_dict[self.pick_variable]}] for """
        if self.pick_toggle == "DatasetID":
            p2 = f""" the datasets {self.pick_dsids} """
        else: #self.pick_toggle == "SAMBA obs.":
            p2 = f""" the region {self.pick_basin} """
        p3 = f"""from {np.datetime_as_string(self.startX, unit='s')} to {np.datetime_as_string(self.endX, unit='s')}"""
        # import pdb; pdb.set_trace();
        p4 = f""" Number of Profiles: {
            self.data_in_view.profile_num.max()-self.data_in_view.profile_num.min()}"""
        self.markdown.object = p1+p2+p3+p4 

        #import pdb; pdb.set_trace();
        return p1+p2+p3+p4

    # empty initialization for use later
    markdown = pn.pane.Markdown("")

    def keep_zoom(self, x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    @param.depends("pick_display_threshold", watch=True)
    def update_display_threshold(self):
        for var in [
            "pick_variable",
            "pick_basin",
            "pick_toggle",
            "pick_dsids",
            "pick_cnorm",
            "pick_aggregation",
            "pick_mld",
            #"pick_mean",
            "pick_TS",
            "pick_profiles",
            "pick_TS_colored_by_variable",
            "pick_contours",
            "pick_high_resolution",
            "button_inflow",
        ]:
            self.param[var].precedence = self.pick_display_threshold

    @param.depends("pick_show_ctrls", watch=True)
    def update_display_threshold(self):
        try:
            # first run, when layout does not exist, this fails deliberately.
            layout[0][0].visible = self.pick_show_ctrls
        except:
            pass

    @param.depends("pick_toggle","pick_basin", watch=True)
    def update_datasource(self):
        # toggles visibility
        if self.pick_toggle == 'DatasetID':
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
        self.endX = np.datetime64(f"2024-01-18")
        self.pick_startX = np.datetime64("2024-01-15")
        self.pick_endX = np.datetime64(f"2024-01-18")

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
        self.pick_variable = "oxygen_concentration"

        return  # self.dynmap*text_annotation

    @param.depends("pick_basin", "pick_dsids", "pick_toggle", watch=True)
    def change_basin(self):
        # bug: setting watch=True enables correct reset of (y-) coordinates, but leads to double initialization (slow)
        # setting watch=False fixes initialization but does not keep y-coordinate.
        if self.pick_toggle == 'SAMBA obs.':
            # first case, , user selected an aggregation, e.g. 'Bornholm Basin'
            meta = metadata[metadata["basin"] == self.pick_basin]
            meta = utils.drop_overlaps_fast(meta)
        else:
            # second case, user selected dids
            meta = metadata.loc[self.pick_dsids]
        # hacky way to differentiate if called via synclink or refreshed with UI buttons
        if not len(meta):
            self.startX = np.datetime64('2021-01-01')
            self.endX = np.datetime64('2024-01-01')
            self.pick_startX = np.datetime64('2021-01-01')
            self.pick_endX = np.datetime64('2024-01-01')
            return
        incoming_link=not(isinstance(self.pick_startX, pd.Timestamp))
        #print('ISINSTANCE', isinstance(self.pick_startX, pd.Timestamp))
        #print('INCOMING VIA LINK:', incoming_link)
        if not incoming_link:
            mintime = meta['time_coverage_start (UTC)'].min()
            maxtime = meta['time_coverage_end (UTC)'].max()
            self.startX, self.endX = (mintime.to_datetime64(), maxtime.to_datetime64())
            self.pick_startX, self.pick_endX = (mintime, maxtime)
        else:
            self.pick_startX, self.pick_endX = (self.pick_startX, self.pick_endX)

        self.startY = None
        self.endY = 12

    @param.depends(
        "pick_cnorm",
        "pick_variable",
        "pick_aggregation",
        "pick_mld",
        #"pick_mean",
        "pick_basin",
        "pick_dsids",
        "pick_toggle",
        "pick_TS",
        "pick_contours",
        "pick_TS_colored_by_variable",
        "pick_high_resolution",
        "pick_profiles",
        "pick_display_threshold",  #'pick_startX', 'pick_endX',
        #watch=True,
    )  # outcommenting this means just depend on all, redraw always
    def create_dynmap(self):

        # self.markdown.object = self.update_markdown()

        self.startX = self.pick_startX
        self.endX = self.pick_endX

        self.startY, self.endY = (self.pick_startY, self.pick_endY)

        #self.startY = self.pick_startY
        #self.endY = self.pick_endY

        # in case coming in over json link
        self.startX = np.datetime64(self.startX)
        self.endX = np.datetime64(self.endX)

        # commonheights = 1000
        x_range = (self.startX, self.endX)
        y_range = (self.startY, self.endY)

        range_stream = RangeXY(x_range=x_range, y_range=y_range).rename()
        range_stream.add_subscriber(self.keep_zoom)
        # range_stream.add_subscriber(self.update_markdown) # Is always one step after, thus deactivated here

        t1 = time.perf_counter()
        pick_cnorm = "linear"

        dmap_raster = hv.DynamicMap(
            self.get_xsection_raster,
            streams=[range_stream],
        )

        if self.pick_aggregation == "mean":
            means = dsh.mean(self.pick_variable)
        if self.pick_aggregation == "std":
            means = dsh.std(self.pick_variable)
        if self.pick_high_resolution:
            pixel_ratio = 1.0
        else:
            pixel_ratio = 0.5
        # if self.pick_aggregation=='var':
        #    means = dsh.var(self.pick_variable)

        if self.pick_TS:
            dmap_TS = hv.DynamicMap(
                self.get_xsection_TS,
                streams=[range_stream],
                cache_size=1,
            )

            dcont = hv.DynamicMap(
                self.get_density_contours, streams=[range_stream]
            ).opts(
                alpha=0.5,
            )
            if not self.pick_TS_colored_by_variable:
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
                    aggregator=means,
                ).opts(
                    cnorm="eq_hist",
                    cmap=dictionaries.cmap_dict[self.pick_variable],
                    #clabel=f"{self.pick_variable}  [{dictionaries.units_dict[self.pick_variable]}]",
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

        dmap = hv.DynamicMap(self.get_xsection, streams=[range_stream], cache_size=1)        
        dmap_rasterized = rasterize(
            dmap_raster,
            aggregator=means,
            # x_sampling=8.64e13/48,
            y_sampling=0.2,
            pixel_ratio=pixel_ratio,
        ).opts(
            # invert_yaxis=True, # Would like to activate this, but breaks the hover tool
            colorbar=True,
            cmap=dictionaries.cmap_dict[self.pick_variable],
            toolbar="above",
            tools=["xwheel_zoom", "reset", "xpan", "ywheel_zoom", "ypan", "hover"],
            default_tools=[],
            # responsive=True, # this currently breaks when activated with MLD
            # width=800,
            # height=commonheights,
            cnorm=self.pick_cnorm,
            active_tools=["xpan", "xwheel_zoom"],
            bgcolor="dimgrey",
            clabel=f"{self.pick_variable}  [{dictionaries.units_dict[self.pick_variable]}]",#self.pick_variable,
        )

        # Here it is important where the xlims are set. If set on rasterized_dmap,
        # zoom limits are kept, if applied in the end zoom limits won't work
        self.dynmap = spread(dmap_rasterized, px=1, how="source").opts(
            # invert_yaxis=True,
            ylim=(self.startY, self.endY),
            )
        if self.pick_contours:
            if self.pick_contours == self.pick_variable:
                self.dynmap = self.dynmap * hv.operation.contours(
                    self.dynmap,
                    levels=10,
                    #group_label='blipp',
                ).opts(
                    # cmap=dictionaries.cmap_dict[self.pick_contours],
                    # group_label='blubb',
                    line_width=2.0,
                ).opts(legend_position='bottom_right',
                       legend_opts={'title':'blubb'})
            else:
                dmap_contour = hv.DynamicMap(
                    self.get_xsection_raster_contour,
                    streams=[range_stream],
                )
                means_contour = dsh.mean(self.pick_contours)
                dmap_contour_rasterized = rasterize(
                    dmap_contour,
                    aggregator=means_contour,
                    y_sampling=0.2,
                    pixel_ratio=pixel_ratio,
                ).opts()
                self.dynmap = self.dynmap * hv.operation.contours(
                    dmap_contour_rasterized,
                    levels=10,
                    #group_label='blipp',
                ).opts(
                    line_width=2.0,
                    # group_label='blubb',
                    # clabel=self.pick_variable,
                ).opts(legend_position='bottom_right',
                       legend_opts={'title':'blubb'})

        if self.pick_mld:
            dmap_mld = hv.DynamicMap(
                self.get_xsection_mld, streams=[range_stream], cache_size=1
            ).opts(responsive=True)
            self.dynmap = (self.dynmap.opts(responsive=True) * dmap_mld.opts(responsive=True)
                ).opts(responsive=True)
            #self.dynmap = (
                #self.dynmap.opts(responsive=True)*dmap_mld).opts(ylim=(self.startY, self.endY),)#, 
                # invert_yaxis=True,) # invert_yaxis=True, # Would like to activate this, but breaks the hover tool) 
                #* dmap_mld.opts(responsive=True)
            #).opts(responsive=True)
        for annotation in self.annotations:
            print("insert text annotations defined in events")
            self.dynmap = self.dynmap * annotation
        if self.pick_TS:
            linked_plots = link_selections(
                self.dynmap.opts(
                    responsive=True
                )
                + dmapTSr.opts(responsive=True, bgcolor="white").opts(
                    padding=(0.05, 0.05)
                ),
                unselected_alpha=0.3,
                cross_filter_mode="overwrite", # could also be union to enable combined selections. More confusing?
            )
            linked_plots.DynamicMap.II = (
                dcont.opts(xlabel="salinity", ylabel="temperature")
                * linked_plots.DynamicMap.II
            )
            return linked_plots
        if self.pick_profiles:
            linked_plots = link_selections(
                self.dynmap.opts(
                    responsive=True
                )
                + dmap_profilesr.opts(
                    responsive=True,
                    bgcolor="white",
                ).opts(
                    padding=(0.05, 0.05),
                ),
                unselected_alpha=0.3,
            )
            linked_plots.DynamicMap.II = linked_plots.DynamicMap.II

            return linked_plots

        else:
            self.dynmap = self.dynmap * dmap.opts(
                # opts.Labels(text_font_size='6pt')
            )
            return self.dynmap.opts(
                responsive=True,
            )



    def create_mean(self):

        self.startX = self.pick_startX
        self.endX = self.pick_endX

        # in case coming in over json link
        self.startX = np.datetime64(self.startX)
        self.endX = np.datetime64(self.endX)
        x_range = (self.startX, self.endX)
        y_range = (self.startY, self.endY)
        range_stream = RangeXY(x_range=x_range, y_range=y_range).rename()
        #dmap_raster = hv.DynamicMap(
        #    self.get_xsection_raster,
        #    streams=[range_stream],
        #)
        dmap = hv.DynamicMap(self.get_xsection, streams=[range_stream], cache_size=1)
        dmap_mean = hv.DynamicMap(
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
        )*dmap#.opts(responsive=True)

        return dmap_mean

    def load_viewport_datasets(self, x_range):
        (x0, x1) = x_range
        dt = x1 - x0
        dtns = dt / np.timedelta64(1, "ns")
        plt_props = {}

        if self.pick_toggle == 'SAMBA obs.':
            # first case, , user selected an aggregation, e.g. 'Bornholm Basin'
            meta = metadata[metadata["basin"] == self.pick_basin]
            meta = utils.drop_overlaps_fast(meta)

            meta = meta[
                # x0 and x1 are the time start and end of our view, the other times
                # are the start and end of the individual datasets. To increase
                # perfomance, datasets are loaded only if visible, so if
                # 1. it starts within our view...
                (
                    (metadata["time_coverage_start (UTC)"] >= x0)
                    & (metadata["time_coverage_start (UTC)"] <= x1)
                )
                |
                # 2. it ends within our view...
                (
                    (metadata["time_coverage_end (UTC)"] >= x0)
                    & (metadata["time_coverage_end (UTC)"] <= x1)
                )
                |
                # 3. it starts before and ends after our view (zoomed in)...
                (
                    (metadata["time_coverage_start (UTC)"] <= x0)
                    & (metadata["time_coverage_end (UTC)"] >= x1)
                )
                |
                # 4. or it both, starts and ends within our view (zoomed out)...
                (
                    (metadata["time_coverage_start (UTC)"] >= x0)
                    & (metadata["time_coverage_end (UTC)"] <= x1)
                )
            ]

        else:
            # second case, user selected dids
            #import pdb; pdb.set_trace();
            meta = metadata.loc[self.pick_dsids]

        # print(f'len of meta is {len(meta)} in load_viewport_datasets')
        if (x1 - x0) > np.timedelta64(720, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 25
        elif (x1 - x0) > np.timedelta64(360, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 10
        elif (x1 - x0) > np.timedelta64(180, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 4
        elif (x1 - x0) > np.timedelta64(90, "D"):
            # activate sparse data mode to speed up reactivity
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 4
            plt_props["subsample_freq"] = 2
        else:
            plt_props["zoomed_out"] = False
            plt_props["dynfontsize"] = 10
            plt_props["subsample_freq"] = 1
        return meta, plt_props

    def get_xsection_mld(self, x_range, y_range):
        try:
            dscopy = utils.add_dive_column(self.data_in_view).compute()
        except:
            dscopy = utils.add_dive_column(self.data_in_view)
        dscopy["depth"] = -dscopy["depth"]
        mld = gt.physics.mixed_layer_depth(
            dscopy.to_xarray(), "temperature", thresh=0.3, verbose=True, ref_depth=5
        )
        gtime = dscopy.reset_index().groupby(by="profile_num").mean().time
        dfmld = (
            pd.DataFrame.from_dict(
                dict(time=gtime.values, mld=-mld.rolling(10, center=True).mean().values)
            )
            .sort_values(by="time")
            .dropna()
        )
        #if len(dfmld) == 0:
        #    import pdb
        #    pdb.set_trace()
        mldscatter = dfmld.hvplot.line(
            x="time",
            y="mld",
            color="white",
            alpha=0.5,
            responsive=True,
        )
        return mldscatter

    def get_xsection_mean(self, x_range, y_range):
        try:
            dscopy = utils.add_dive_column(self.data_in_view).compute()
        except:
            dscopy = utils.add_dive_column(self.data_in_view)
        #dscopy["depth"] = -dscopy["depth"]
        #mld = gt.physics.mixed_layer_depth(
        #    dscopy.to_xarray(), "temperature", thresh=0.3, verbose=True, ref_depth=5
        #)
        if self.pick_aggregation_method == 'mean':
            groups = dscopy.reset_index()[['time', self.pick_variable, 'profile_num']].groupby(by="profile_num").mean()#.time
        elif self.pick_aggregation_method == 'max':
            groups = dscopy.reset_index()[['time', self.pick_variable, 'profile_num']].groupby(by="profile_num").max()#.time
        elif self.pick_aggregation_method == 'min':
            groups = dscopy.reset_index()[['time', self.pick_variable, 'profile_num']].groupby(by="profile_num").min()#.time
        #elif self.pick_aggregation_method == 'std':
        #    groups = dscopy.reset_index()[['time', self.pick_variable, 'profile_num']].groupby(by="profile_num").std()#.time

        gtime = groups.time
        gmean = groups[self.pick_variable]
        #gtmean = dscopy.reset_index().groupby(by="profile_num")[self.pick_variable].mean()
#mld=-mld.rolling(10, center=True).mean().values
        dfmean = (
            pd.DataFrame.from_dict(
                dict(time=gtime.values, mean=gmean.values)
            )
            .sort_values(by="time")
            .dropna()
        )#.rolling(window=4).mean()
        dfmean['mean'] = dfmean['mean'].rolling(4, center=True).mean().values

        meanline = dfmean.hvplot.line(
            x="time",
            y="mean",
            responsive=True,
        )

        return meanline

    def get_xsection_raster(self, x_range, y_range, contour_variable=None):
        (x0, x1) = x_range
        #try:
        #    self.pick_startX = pd.to_datetime(x0)  # setters
        #    self.pick_endX = pd.to_datetime(x1)
        #except:
        #    import pdb; pdb.set_trace();
        #t1 = time.perf_counter()
        #print("start raster")
        self.pick_startX = pd.to_datetime(x0)  # setters
        self.pick_endX = pd.to_datetime(x1)
        meta, plt_props = self.load_viewport_datasets(x_range)
        plotslist1 = []

        if plt_props["zoomed_out"]:
            metakeys = [element.replace("nrt", "delayed") for element in meta.index]
        else:
            metakeys = [
                (
                    element.replace("nrt", "delayed")
                    if element.replace("nrt", "delayed") in all_datasets.index
                    else element
                )
                for element in meta.index
            ]
        if contour_variable:
            variable = contour_variable
        else:
            variable = self.pick_variable
        varlist = []
        for dsid in metakeys:
            ds = dsdict[dsid]
            # import pdb; pdb.set_trace();
            ds = ds[ds.profile_num % plt_props["subsample_freq"] == 0]
            varlist.append(ds)
        # import pdb; pdb.set_trace();
        #if self.pick_mld or self.pick_mean:
        #    'VOTO CONCATTT'
        varlist = utils.voto_concat_datasets(varlist)
        if varlist:
            # concat and drop_duplicates could potentially be done by pandarallel
            if self.pick_TS:
                nanosecond_iterator = 1
                for ndataset in varlist:
                    ndataset.index = ndataset.index + np.timedelta64(
                        nanosecond_iterator, "ns"
                    )
                    nanosecond_iterator += 1
            dsconc = dd.concat(varlist)
            dsconc = dsconc.loc[x_range[0] : x_range[1]]
            # could be parallelized
            if self.pick_TS:
                try:
                    dsconc = dsconc.drop_duplicates(
                        subset=["temperature", "salinity"]
                    ).compute()
                except:
                    dsconc = dsconc.drop_duplicates(subset=["temperature", "salinity"])
            self.data_in_view = dsconc
            self.update_markdown(x_range, y_range)

            mplt = create_single_ds_plot_raster(data=dsconc, variable=variable)
            #t2 = time.perf_counter()
            #print(t2 - t1)
            return mplt
        else:
            #data = {"time":[], "depth":[], variable:[]}  # Declaration line
            #data = pd.DataFrame.from_dict(data)
            #raster = data.hvplot.points(
            #    x="time",
            #    y="depth",
            #    c=variable,
            #)
            return self.create_None_element("Overlay")

    def get_xsection_raster_contour(self, x_range, y_range):
        # This function exists because I cannot pass variables directly
        variable = self.pick_contours
        return self.get_xsection_raster(x_range, y_range, contour_variable=variable)

    def get_xsection_TS(self, x_range, y_range):
        dsconc = self.data_in_view
        t1 = time.perf_counter()
        thresh = dsconc[["temperature", "salinity"]].quantile(q=[0.001, 0.999])
        t2 = time.perf_counter()
        mplt = dsconc.hvplot.scatter(
            x="salinity",
            y="temperature",
            c=self.pick_variable,
        )[
            thresh["salinity"].iloc[0] - 0.5 : thresh["salinity"].iloc[1] + 0.5,
            thresh["temperature"].iloc[0] - 0.5 : thresh["temperature"].iloc[1] + 0.5,
        ]

        return mplt

    def get_xsection_profiles(self, x_range, y_range):
        dsconc = self.data_in_view
        t1 = time.perf_counter()
        thresh = dsconc[self.pick_variable].quantile(q=[0.001, 0.999])
        t2 = time.perf_counter()
        try:
            thresh = thresh.compute()  # .iloc[0]
        except:
            thresh = thresh
        mplt = dsconc.hvplot.scatter(
            x=self.pick_variable,
            y="depth",
            # No clue if this was good or bad. Needs to be testeded!
            c=self.pick_variable,
        )  # [thresh.iloc[0]-(0.1*thresh.iloc[0]):thresh.iloc[1]+(0.1*thresh.iloc[1])]
        # [thresh.iloc[0]-(0.1*thresh.iloc[0]):thresh.iloc[1]+(0.1*thresh.iloc[1])]#,
        # thresh['temperature'].iloc[0]-0.5:thresh['temperature'].iloc[1]+0.5]

        return mplt

    def get_density_contours(self, x_range, y_range):
        # for the TS plot
        import gsw

        dsconc = self.data_in_view
        t1 = time.perf_counter()
        thresh = dsconc[["temperature", "salinity"]].quantile(q=[0.001, 0.999])

        try:
            thresh = thresh.compute()  # .iloc[0]
        except:
            thresh = thresh

        smin, smax = (thresh["salinity"].iloc[0] - 1, thresh["salinity"].iloc[1] + 1)
        tmin, tmax = (
            thresh["temperature"].iloc[0] - 1,
            thresh["temperature"].iloc[1] + 1,
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
        )
        # this is good but the ranges are not yet automatically adjusted.
        # also, maybe the contour color should be something more discrete
        return dcont


    def create_None_element(self, type):
        # This is just a hack because I can't return None to dynamic maps
        if type == "Overlay":
            element = hv.Overlay(
                hv.HLine(0).opts(color="black", alpha=0.1)
                * hv.HLine(0).opts(color="black", alpha=0.1)
                #* hv.Text(
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
        t1 = time.perf_counter()
        meta, plt_props = self.load_viewport_datasets(x_range)

        meta_start_in_view = meta[(meta["time_coverage_start (UTC)"] > x0)]
        meta_end_in_view = meta[(meta["time_coverage_end (UTC)"] < x1)]

        startvlines = (
            hv.VLines(meta_start_in_view["time_coverage_start (UTC)"])
            .opts(color="grey", line_width=1)#, spike_length=20)
            #.opts(position=-10)
        )
        endvlines = (
            hv.VLines(meta_end_in_view["time_coverage_end (UTC)"])
            .opts(color="grey", line_width=1)#, spike_length=20)
            #.opts(position=-10)
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
                time=meta_start_in_view["time_coverage_start (UTC)"].values,
                y=5,
                text=meta_start_in_view.index.str.replace("nrt_", ""),
            )
        )
        ds_labels = hv.Labels(data).opts(
            fontsize=12, text_align="left"  # plt_props['dynfontsize'],
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
            return self.create_None_element("Overlay")


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
        dfm = all_metadata.sort_values(
            "basin"
        )
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


def create_app_instance():
    glider_dashboard = GliderDashboard()
    # glider_explorer2=GliderExplorer()

    meta_dashboard = MetaDashboard()

    # Data options
    ctrl_data = pn.Column( # top stack, dataset and basin options
        'Choose input data either based on basin location or ID',
        pn.Param(
            glider_dashboard,
            parameters=["pick_toggle"],
            widgets={'pick_toggle':pn.widgets.RadioButtonGroup, 'button_type': 'success'},
            #css_classes=["widget-button"],
            #default_layout=pn.Column,
            show_name=False,
            #width=100,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_basin"],
            #widgets={'pick_basin':pn.widgets.MultiChoice(max_items=1)}
            default_layout=pn.Column,
            #max_items=1,
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_dsids"],
            widgets={'pick_dsids':pn.widgets.MultiChoice},
            show_name=False,
        ),
        #styles={"background": "#C0C0C0"},
    )

    # contour plot options
    ctrl_contour = pn.Column(
        pn.Param(
            glider_dashboard,
            parameters=["pick_variable"],
            default_layout=pn.Column,
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_cnorm"],
            widgets={'pick_cnorm': pn.widgets.RadioButtonGroup},
            show_name=False,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_aggregation"],
            widgets={'pick_aggregation': pn.widgets.RadioButtonGroup},
            show_name=False,
            show_labels=True,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_contours"],
            show_name=False,
        ),
        #styles={"background": "#f0f0f0"},
    )

    # scatter options
    ctrl_scatter = pn.Column(
        pn.Param(
            glider_dashboard,
            parameters=["pick_TS", "pick_TS_colored_by_variable"],
            default_layout=pn.Row,
            show_name=False,
            # display_threshold=10,
        ),
        pn.Param(
            glider_dashboard,
            parameters=["pick_profiles"],
            show_name=False,
            # display_threshold=10,
        ),
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
        #pn.Param(
        #    glider_dashboard,
        #    parameters=["pick_mean"],
        #    show_name=False,
        #    # display_threshold=0.5,
        #),
        #pn.Param(
        #    glider_dashboard,
        #    parameters=["button_inflow"],
        #    show_name=False,
        #    # display_threshold=10,
        #),
        #button_cols,
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

    pick_aggregation_method = pn.Param(
            glider_dashboard,
            parameters=["pick_aggregation_method"],
            widgets={'pick_aggregation_method': pn.widgets.RadioButtonGroup},
            show_name=False,
            show_labels=True,)

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
        contentcolumn.height=1050

    def remove_column(hex_id=None):
        """
        Dynamically remove a column from the app.
        """
        # value = random.randint(0, 100)
        # column = pn.widgets.TextInput(name="Enter a number", value=str(value))
        contentcolumn.remove(meancolumn)
        contentcolumn.height=500

    add_row = pn.widgets.Button(name="Add aggregation row")
    clear_rows = pn.widgets.Button(name="Clear additional rows")

    #main = pn.Column("# Dynamically add new rows", button_cols, layout)

    # Add interactivity
    clear_rows.on_click(lambda _: remove_column())
    add_row.on_click(lambda _: create_column())

    # this keeps the url in sync with the parameter choices and vice versa
    if pn.state.location:
        pn.state.location.sync(
            glider_dashboard,
            {
                "pick_basin": "pick_basin",
                "pick_dsids": "pick_dsids",
                "pick_toggle": "pick_toggle",
                "pick_show_ctrls": "pick_show_ctrls",
                "pick_variable": "pick_variable",
                "pick_aggregation": "pick_aggregation",
                "pick_aggregation_method": "pick_aggregation_method",
                "pick_mld": "pick_mld",
                #"pick_mean": "pick_mean",
                "pick_cnorm": "pick_cnorm",
                "pick_TS": "pick_TS",
                "pick_profiles": "pick_profiles",
                "pick_TS_colored_by_variable": "pick_TS_colored_by_variable",
                "pick_contours": "pick_contours",
                "pick_high_resolution": "pick_high_resolution",
                "pick_startX": "pick_startX",
                "pick_endX": "pick_endX",
                "pick_startY": "pick_startY",
                "pick_endY": "pick_endY",
                "pick_display_threshold": "pick_display_threshold",
                "pick_contour_heigth": "pick_contour_heigth",
            },
        )

    contentcolumn = pn.Column(
        #pn.Row(
        glider_dashboard.create_dynmap,
            #glider_dashboard.create_mean,
            pn.Param(
            glider_dashboard,
            parameters=["pick_show_ctrls"],
            show_name=False,),
            height=glider_dashboard.pick_contour_heigth,
            #),
        #pn.Row("# Add data aggregations (mean, max, std...)", button_cols),
    )

    layout = pn.Column(
        pn.Row( # row with controls, trajectory plot and TS plot
            pn.Accordion(
                toggle=True,
                objects=[('Choose dataset(s)', ctrl_data),
                ('Contour plot options', ctrl_contour),
                ('Linked (scatter-)plots', ctrl_scatter),
                ('Aggregations (WIP)', pn.Column(
                    pick_aggregation_method,
                    add_row, 
                    clear_rows, 
                    )),
                ('more', ctrl_more),
                #('WIP',add_row),
                ],),
                contentcolumn,
                #, pn.Row(button_cols)])],
                visible=True,
            #height=500,
        ),
        pn.Row(glider_dashboard.markdown),
        pn.Row(
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
        ),
        #pn.Row("# Dynamically add new rows", button_cols)
        # visible=False, # works, but hides everything!
    )

    # it is necessary to hide the controls as a very last option, because hidden controls cannot be accessed as variables
    # in the control flow above. So hiding the controls earlier "defaults" all url and manual settings. 
    if glider_dashboard.pick_show_ctrls == False:
        layout[0][0].visible = glider_dashboard.pick_show_ctrls
    return layout


# usefull to create secondary plot, but not fully indepentently working yet:
# glider_explorer2=GliderExplorer()
layout = create_app_instance()
# app = layout # create_app_instance()
# app2 = create_app_instance()
layout.servable()
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
