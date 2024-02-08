import xarray
import glidertools as gt
#import hvplot.dask
#import hvplot.xarray
import hvplot.pandas
import cmocean
import holoviews as hv
import pathlib
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from bokeh.models import DatetimeTickFormatter, HoverTool
from holoviews.operation import decimate
from holoviews.streams import RangeX
import numpy as np
from functools import reduce
import panel as pn
import param
import datashader.transfer_functions as tf
import time
import plotly.express as px

from download_glider_data import utils as dutils
import utils
import dictionaries
import pickle

pn.extension('plotly')

# unused imports
# import hvplot.pandas
#import cudf # works w. cuda, but slow.
try:
    import hvplot.cudf
except:
    print('no cudf available, that is fine but slower')

# all metadata exists for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
"""
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, all_datasets) # hacky

###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration', 'cdom', 'backscatter_scaled', 'longitude']
dsdict = dutils.download_glider_dataset(all_dataset_ids, metadata,
                                        variables=variables) """
file = open('cached_data_dictionary.pickle', 'rb')
dsdict = pickle.load(file)
file.close()
#import pdb; pdb.set_trace();

####### specify global plot variables ####################
#df.index = cudf.to_datetime(df.index)
text_opts  = hv.opts.Text(text_align='left', text_color='black') #OOOOOOOOOOOOOOO
ropts = dict(
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
             default_tools=[],
             active_tools=['xpan', 'xwheel_zoom'],
             bgcolor="dimgrey",
             ylim=(-8,None)
            )

def plot_limits(plot, element):
    plot.handles['x_range'].min_interval = np.timedelta64(2, 'h')
    plot.handles['x_range'].max_interval = np.timedelta64(int(5*3.15e7), 's') # 5 years
    plot.handles['y_range'].min_interval = 10
    plot.handles['y_range'].max_interval = 500

def create_single_ds_plot(data, metadata, variable, dsid, plt_props):
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(**ropts).opts(text_opts)

    startvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_start (UTC)']).opts(color='grey', line_width=1)
    endvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_end (UTC)']).opts(color='grey', line_width=1)
    return text_annotation*startvline*endvline


def create_single_ds_plot_raster(
        data):
    t1 = time.perf_counter()
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        )
    #adjscatter = hv.operation.Scatter(data, dimension='cplotvar')
    t2 = time.perf_counter()
    return raster #<< adjscatter


def load_viewport_datasets(x_range):
    t1 = time.perf_counter()
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    plt_props = {}
    meta = metadata[metadata['basin']==currentobject.pick_basin]
    meta = meta[
            # x0 and x1 are the time start and end of our view, the other times
            # are the start and end of the individual datasets. To increase
            # perfomance, datasets are loaded only if visible, so if
            # 1. it starts within our view...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x1)) |
            # 2. it ends within our view...
            ((pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1)) |
            # 3. it starts before and ends after our view (zoomed in)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x1)) |
            # 4. or it both, starts and ends within our view (zoomed out)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1))
            ]

    print(f'len of meta is {len(meta)} in load_viewport_datasets')
    if (x1-x0)>np.timedelta64(360, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=50
    elif (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=20
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    else:
        # load delayed mode datasets for more detail
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=1
    t2 = time.perf_counter()
    return meta, plt_props


def get_xsection(x_range):
    #import pdb; pdb.set_trace()
    #(x0, x1) = x_range
    t1 = time.perf_counter()
    variable='temperature'
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist = []
    for dsid in meta.index:
        # this is just plotting lines and meta, no need for 'delayed' data (?)
        data=dsdict[dsid]
        single_plot = create_single_ds_plot(
            data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    t2 = time.perf_counter()
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_mld(x_range):
    t1 = time.perf_counter()
    variable='temperature'
    meta, plt_props = load_viewport_datasets(x_range)
    # activate this for high delayed resolution
    # metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
    metakeys = meta.index
    varlist = [dsdict[dsid] for dsid in metakeys]
    dslist = utils.voto_concat_datasets(varlist)
    dslist = [utils.add_dive_column(ds) for ds in dslist]
    plotslist = []
    for ds in dslist:
        mld = gt.physics.mixed_layer_depth(ds.to_xarray(), 'temperature', thresh=0.1, verbose=False, ref_depth=10)
        gtime = ds.reset_index().groupby(by='profile_num').mean().time
        #gt.utils.group_by_profiles(ds, variables=['time', 'temperature']).mean().time.values
        gmld = mld.values
        dfmld = pd.DataFrame.from_dict(dict(time=gtime, mld=gmld))
        #dfmld['mld'] = dfmld.mld.rolling(window=10, min_periods=5, center=True).mean()
        mldscatter = dfmld.hvplot.line(
            x='time',
            y='mld',
            color='white',
            alpha=0.5,
        )
        plotslist.append(mldscatter)
    t2 = time.perf_counter()
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_raster(x_range):
    (x0, x1) = x_range
    GliderExplorer.xmin, GliderExplorer.xmax = x_range
    #global x_min_global
    #global x_max_global
    #x_min_global = x0
    #x_max_global = x1
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist1 = []
    #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
    # activate this for high res data
    if plt_props['zoomed_out']:
        metakeys = [element.replace('nrt', 'delayed') for element in meta.index]
    else:
        metakeys = [element.replace('nrt', 'delayed') if
            element.replace('nrt', 'delayed') in all_datasets.index else
            element for element in meta.index]

    varlist = [dsdict[dsid] for dsid in metakeys]
    dsconc = pd.concat(varlist)
    # import pdb; pdb.set_trace();

    dsconc['cplotvar'] = dsconc[currentobject.pick_variable]
    dsconc = dsconc.iloc[0:-1:plt_props['subsample_freq']]
    # import pdb; pdb.set_trace();
    mplt = create_single_ds_plot_raster(data=dsconc)
    t2 = time.perf_counter()
    return mplt


def get_xsection_TS(x_range):
    #(x0, x1) = x_range
    #global x_min_global
    #global x_max_global
    #x_min_global = x0
    #x_max_global = x1
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist1 = []
    #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
    # activate this for high res data
    if plt_props['zoomed_out']:
        metakeys = [element.replace('nrt', 'delayed') for element in meta.index]
    else:
        metakeys = [element.replace('nrt', 'delayed') if
            element.replace('nrt', 'delayed') in all_datasets.index else
            element for element in meta.index]

    varlist = [dsdict[dsid] for dsid in metakeys]
    dsconc = pd.concat(varlist)
    # import pdb; pdb.set_trace();

    #dsconc['cplotvar'] = dsconc[currentobject.pick_variable]
    #dsconc = dsconc.iloc[0:-1:plt_props['subsample_freq']]
    # import pdb; pdb.set_trace();
    #mplt = create_single_ds_plot_raster(data=dsconc)
    #t2 = time.perf_counter()
    mplt = dsconc.hvplot.scatter(
        x='salinity',
        y='temperature',
        #c='cplotvar',
        )
    return mplt


def get_xsection_points(x_range):
    # currently not activated, but almost completely working.
    # only had some slight problems to keep zoom settings on variable change,
    # but that should be easy to solve...
    (x0, x1) = x_range

    if (x1-x0)<np.timedelta64(14, 'D'):
        meta, plt_props = load_viewport_datasets(x_range)
        plotslist1 = []
        #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
        metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
        varlist = [dsdict[dsid] for dsid in metakeys]
        dsconc = pd.concat(varlist)
        dsconc['cplotvar'] = dsconc[currentobject.pick_variable]
        points = dsconc.hvplot.points(
            x='time',
            y='depth',
            c='cplotvar',
            )
    else:
        dsconc = pd.DataFrame.from_dict(
            dict(time=[x0],
                 depth=[0],
                 cplotvar=[np.nan]))
        points = dsconc.hvplot.points(
            x='time',
            y='depth',
            c='cplotvar',
            hover_cols=['Value'],
            )
    return points



class GliderExplorer(param.Parameterized):

    pick_variable = param.ObjectSelector(
        default='temperature', objects=[
        'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration', 'cdom', 'backscatter_scaled'],
        label='variable', doc='Variable presented as colormesh')
    pick_basin = param.ObjectSelector(
        default='Bornholm Basin', objects=[
        'Bornholm Basin', 'Eastern Gotland',
        'Western Gotland', 'Skagerrak, Kattegat',
        'Åland Sea'], label='SAMBA observatory'
    )
    pick_cnorm = param.ObjectSelector(
        default='linear', objects=['linear', 'eq_hist', 'log'], doc='colorbar transformations', label='cbar scale')
    pick_aggregation = param.ObjectSelector(
        default='mean', objects=['mean', 'std', 'var'], label='aggregation',
        doc='choose method to aggregate different values that fall into one bin')
    pick_mld = param.Boolean(
        default=False, label='MLD', doc='mixed layer depth')
    pick_TS = param.Boolean(
        default=False, label='TSplot', doc='activate salinity temperature diagram')
    #button_inflow = param..Button(name='Tell me about inflows', icon='caret-right', button_type='primary')
    # create a button that when pushed triggers 'button'
    button_inflow = param.Action(lambda x: x.param.trigger('button_inflow'), label='Show animation with labels!')

    #dynmap = None
    # on initial load, show all data
    #xmin =
    x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
        metadata['time_coverage_end (UTC)'].max().to_datetime64())

    #global x_min_global
    #global x_max_global
    #x_min_global, x_max_global = x_range
    xmin, xmax = x_range

    #x_range=(x_min_global,
    #         x_max_global)
    range_stream = RangeX(x_range=x_range)
    annotations = []
    about = """\
    # About
    This is designed to visualize data from the Voice of the Ocean SAMBA observatories. For additional datasets, visit observations.voiceoftheocean.org.
    """
    markdown = pn.pane.Markdown(about)

    @param.depends('button_inflow', watch=True)
    def execute_event(self):
        self.markdown.object = """\
        # Baltic Inflows
        Baltic Inflows are transporting salt and oxygen into the depth of the Baltic Sea.
        """
        #global x_min_global
        #global x_max_global
        #x_min_global = np.datetime64('2023-12-01')
        #x_max_global = np.datetime64('2023-12-14')
        self.pick_variable = 'temperature'
        time.sleep(5)
        print('event:plot reloaded')
        text_annotation = hv.Text(
            x=np.datetime64('2023-12-07'),
            y=20, text='Look at this cool inflow!',
            fontsize=10,
            )
        self.xmin = np.datetime64('2023-12-01')
        self.xmax = np.datetime64('2023-12-14')
        self.annotations.append(text_annotation)
        self.pick_variable = 'oxygen_concentration'

        return #self.dynmap*text_annotation

    @param.depends('pick_basin', watch=True)
    def change_basin(self):
        # on initial load, show all data
        print('basin changed!!!', self.pick_basin)
        x_range=(
        metadata[metadata['basin']==self.pick_basin]['time_coverage_start (UTC)'].min().to_datetime64(),
        metadata[metadata['basin']==self.pick_basin]['time_coverage_end (UTC)'].max().to_datetime64())
        self.x_min_global, self.x_max_global = x_range

    @param.depends('pick_cnorm','pick_variable', 'pick_aggregation',
        'pick_mld', 'pick_basin', 'pick_TS') # outcommenting this means just depend on all, redraw always
    def create_dynmap(self):
        x_range=(self.xmin,
                 self.xmax)
        range_stream = RangeX(x_range=x_range)
        global currentobject
        currentobject = self
        t1 = time.perf_counter()
        pick_cnorm='linear'

        dmap_raster = hv.DynamicMap(
            get_xsection_raster,
            streams=[range_stream],
            #cache_size=1,)
        )
        self.dynmap_raster = dmap_raster


        if self.pick_aggregation=='mean':
            means = dsh.mean('cplotvar')
        if self.pick_aggregation=='std':
            means = dsh.std('cplotvar')
        if self.pick_aggregation=='var':
            means = dsh.var('cplotvar')

        if self.pick_TS:
            dmap_TS = hv.DynamicMap(
                get_xsection_TS,
                streams=[range_stream],
                #cache_size=1,)
            )
            dmapTSr = rasterize(dmap_TS).opts(
                cnorm='eq_hist',
                height=400,)

        #import pdb; pdb.set_trace()

        # adjoint
        #dmap_rasterized = dmap_rasterized #* adjoint

        dmap = hv.DynamicMap(
            get_xsection,
            streams=[range_stream],
            cache_size=1)
        t2 = time.perf_counter()

        dmap_points = hv.DynamicMap(
            get_xsection_points,
            streams=[range_stream],
            cache_size=1
            )
        dmap_points = spread(datashade(
            dmap_points,
            aggregator=means,
            cnorm=self.pick_cnorm,
            cmap=dictionaries.cmap_dict[self.pick_variable],), px=4).opts(
                invert_yaxis=True,
                toolbar='above',
                tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
                default_tools=[],
                width=800,
                height=400,
                active_tools=['xpan', 'xwheel_zoom'],
                bgcolor="dimgrey",)
        if self.pick_mld:
            dmap_mld = hv.DynamicMap(get_xsection_mld, streams=[range_stream], cache_size=1)
        #if self.pick_mld:
            #return (dmap_rasterized*dmap_points).opts(xlim=(x_min_global, x_max_global))*dmap*dmap_mld
            #dynmap = (dmap_points*dmap_rasterized.opts(
                #xlim=(x_min_global, x_max_global)
            #    ))*dmap*dmap_mld.opts(
                    #xlim=(x_min_global, x_max_global)
                    #ylim=(-8,None)
            #        )
            #self.dynmap = dynmap
            #pass
            #dynmap = (dmap_points*dmap_rasterized*dmap_mld).opts(
                #xlim=(x_min_global, x_max_global)
            #    )*dmap.opts(
            #        ylim=(-8,None)
            #        )
        dmap_rasterized = rasterize(dmap_raster,
            aggregator=means,
            #x_sampling=8.64e13/24,
            y_sampling=.2,
            #invert_yaxis=True,
            ).opts(
            #alpha=0.2,
            colorbar=True,
            cmap=dictionaries.cmap_dict[self.pick_variable],#,cmap
            toolbar='above',
            tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],#, 'hover'],
            default_tools=[],
            # ylim=(0,90),
            #responsive=True,
            width=800,
            height=400,
            cnorm=self.pick_cnorm,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",
            clabel=self.pick_variable)
        #adjoint = dmap_rasterized.hist()
        #dynmap = dmap_rasterized.hist().opts(
        #        xlim=(self.xmin, self.xmax))
        dynmap = (dmap_rasterized*dmap_points*dmap).opts(
                xlim=(self.xmin, self.xmax)).opts(hooks=[plot_limits])
        if self.pick_mld:
            dynmap = dynmap * dmap_mld
        if self.pick_TS:
            dynmap = dynmap + dmapTSr
        for annotation in self.annotations:
            print('insert text annotations defined in events')
            dynmap = dynmap*annotation
        #import pdb; pdb.set_trace()
        #dynmap.handles['x_range'].min_interval = np.timedelta64(2, 'h')
        #dynmap.handles['x_range'].max_interval = np.timedelta64(int(5*3.15e7), 's') # 5 years
        #dynmap.handles['y_range'].min_interval = 10
        #dynmap.handles['y_range'].max_interval = 500

        return dynmap
                #*None
        #dynmap = (dmap_points*dmap_rasterized).opts(
                #xlim=(x_min_global, x_max_global)
        #        )*dmap.opts(
        #            ylim=(-8,None)
        #            )
            #return (dmap_rasterized*dmap_points).opts(xlim=(x_min_global, x_max_global))*dmap
        #    self.dynmap = dynmap


        # print(x_min_global)
        #if self.pick_TS:
        #    return dynmap.opts(xlim=(x_min_global, x_max_global))+dmapTSr#*adjoint)#, dynmap
        #else:
        #    if self.pick_mld:
        #        return dynmap.opts(xlim=(x_min_global, x_max_global))*dmap_mld.opts(xlim=(x_min_global, x_max_global))#*adjoint
        #    else:
        #        return dynmap.opts(xlim=(x_min_global, x_max_global))#*adjoint
        #return dmap*dmap_mld
        #ToDO: restore keep zoom functionality


class MetaExplorer(param.Parameterized):
    pick_serial = param.ObjectSelector(
        default='glider_serial', objects=[
        'glider_serial', 'optics_serial', 'altimeter_serial',
        'irradiance_serial','project',],
        label='Equipment Ser. No.', doc='Track equipment or gliders')

    @param.depends('pick_serial') # outcommenting this means just depend on all, redraw always
    def create_timeline(self):
        dfm = all_metadata.sort_values('basin')#px.data.iris() # replace with your own data source
        #fig = make_subplots(rows=1, cols=1,
        #                shared_xaxes=True,
        #                vertical_spacing=0.02)
        dims=self.pick_serial
        fig = px.timeline(dfm,
            x_start="time_coverage_start (UTC)",
            x_end="time_coverage_end (UTC)",
            y="basin",
            hover_name=dfm.index,
            #color_discrete_map=['lightgrey'],
            color_discrete_map={
            0: "lightgrey", "nan":"grey"},
            hover_data=['ctd_serial', 'optics_serial'],
            color=dims,
            pattern_shape=dims,
            width=1000, height=400,
            #scrollZoom=True,
                    )

            # Add range slider
        fig.update_layout(
            title=dims,
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                #type="date"
            )
        )


        # fig.update_layout(yaxis=dict(autorange="reversed"))
        #fig.update_layout(barmode='stack', yaxis={'categoryorder':'total descending'})
        #fig.update_layout(barmode='group')
        for shape in fig['data']:
            shape['opacity'] = 0.7
        #for shape in fig['data']:
            #shape['opacity'] = 0.7
        for i, d in enumerate(fig.data):
            d.width = (metadata.deployment_id%2+10)/12
        return fig
        #config = {'scrollZoom': True}
        #import pdb; pdb.set_trace()
        #return timeline_fig

def create_app_instance():
    glider_explorer=GliderExplorer()
    meta_explorer=MetaExplorer()
    layout = pn.Column(
    pn.Row(
        glider_explorer.param,
        glider_explorer.create_dynmap),
    pn.Row(glider_explorer.markdown),
    pn.Row(
        meta_explorer.param),
    pn.Row(
        meta_explorer.create_timeline))
    return layout

# usefull to create secondary plot, but not fully indepentently working yet:
# glider_explorer2=GliderExplorer()

app = create_app_instance()
app.servable()
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
* add secondary plot or the option for secondary linked plot
* disentangle interactivity, so that partial refreshes (e.g. mixed layer calculation only) don't trigger complete refresh
* otpimal colorbar range (percentiles?)
* on selection of a new basin, I should reset the ranges. Otherwise it could come up with an error when changing while having unavailable x_range.
...
"""
