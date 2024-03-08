import time
#import xarray
import glidertools as gt
import hvplot.dask
#import hvplot.xarray
import hvplot.pandas
import cmocean
import holoviews as hv
from holoviews import opts
#import pathlib
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import rasterize, spread, dynspread, regrid
from holoviews.selection import link_selections
#from bokeh.models import DatetimeTickFormatter, HoverTool
#from holoviews.operation import decimate
from holoviews.streams import RangeX, RangeXY
import numpy as np
from functools import reduce
import panel as pn
import param
#import datashader.transfer_functions as tf
import plotly.express as px
#import warnings
#import pickle
import initialize
import dask
import dask.dataframe as dd
#import asyncio
#from bokeh.models import CrosshairTool, Span

from download_glider_data import utils as dutils
import utils
import dictionaries

pn.extension('plotly')
#pn.param.ParamMethod.loading_indicator = True

# unused imports
try:
    # cudf support works, but is currently not faster
    import hvplot.cudf
except:
    print('no cudf available, that is fine but slower')

# all metadata exists for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
metadata['time_coverage_start (UTC)'] = metadata['time_coverage_start (UTC)'].dt.tz_convert(None)
metadata['time_coverage_end (UTC)'] = metadata['time_coverage_end (UTC)'].dt.tz_convert(None)

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
#file = open('cached_data_dictionary.pickle', 'rb')
dsdict = initialize.dsdict
#file.close()
#import pdb; pdb.set_trace();

####### specify global plot variables ####################
#df.index = cudf.to_datetime(df.index)
text_opts  = hv.opts.Text(text_align='left', text_color='black') #OOOOOOOOOOOOOOO
ropts = dict(
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
             default_tools=[],
             active_tools=['xpan', 'xwheel_zoom'],
             bgcolor="dimgrey",
             #ylim=(-8,None)
            )

def plot_limits(plot, element):
    plot.handles['x_range'].min_interval = np.timedelta64(2, 'h')
    plot.handles['x_range'].max_interval = np.timedelta64(int(5*3.15e7), 's') # 5 years
    plot.handles['y_range'].min_interval = 10
    plot.handles['y_range'].max_interval = 500

"""
def create_single_ds_plot(metadata, variable, dsid, plt_props, x_range):
    # return create_None_element()
    x0, x1 = x_range
    elements = []
    if metadata.loc[dsid]['time_coverage_start (UTC)']>x0:
        text_annotation = hv.Text(
            x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
            y=-2,
            text=dsid.replace('nrt_', ''),
            fontsize=plt_props['dynfontsize'],
                ).opts(**ropts).opts(text_opts)
        startvline = hv.VLine(metadata.loc[dsid][
            'time_coverage_start (UTC)']).opts(color='grey', line_width=1)
        elements.append(text_annotation)
        elements.append(startvline)
    if metadata.loc[dsid]['time_coverage_end (UTC)']<x1:
        endvline = hv.VLine(metadata.loc[dsid][
            'time_coverage_end (UTC)']).opts(color='grey', line_width=1)
        elements.append(endvline)
        # This text annotation '.' is preventing a bug when zooming, e.g. turning
        # the Vline object into an overlay object and thus making dmaps compatible
        # later on. Test zooming if remove.
        text_annotation = hv.Text(
            x=metadata.loc[dsid]['time_coverage_end (UTC)'] ,
            y=-2,
            text='.',
            fontsize=plt_props['dynfontsize'],
                ).opts(**ropts).opts(text_opts)
        elements.append(text_annotation)
    if elements:
        return reduce(lambda x, y: x*y, elements)
    else:
        return create_None_element('')
        #.opts(xlim=x_range)#(text_annotation*startvline*endvline)#.opts(xlim=(GliderExplorer.startX, GliderExplorer.endX))
"""

def create_single_ds_plot_raster(
        data, variable):
    # https://stackoverflow.com/questions/32318751/holoviews-how-to-plot-dataframe-with-time-index
    raster = data.hvplot.points(
        x='time',
        y='depth',
        c=variable,
        )
    return raster

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
            ((metadata['time_coverage_start (UTC)']>=x0) &
            (metadata['time_coverage_start (UTC)']<=x1)) |
            # 2. it ends within our view...
            ((metadata['time_coverage_end (UTC)']>=x0) &
            (metadata['time_coverage_end (UTC)']<=x1)) |
            # 3. it starts before and ends after our view (zoomed in)...
            ((metadata['time_coverage_start (UTC)']<=x0) &
            (metadata['time_coverage_end (UTC)']>=x1)) |
            # 4. or it both, starts and ends within our view (zoomed out)...
            ((metadata['time_coverage_start (UTC)']>=x0) &
            (metadata['time_coverage_end (UTC)']<=x1))
            ]

    #print(f'len of meta is {len(meta)} in load_viewport_datasets')
    if (x1-x0)>np.timedelta64(720, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=25
    elif (x1-x0)>np.timedelta64(360, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=10
    elif (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=4
    elif (x1-x0)>np.timedelta64(90, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=2
    else:
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=1
    t2 = time.perf_counter()
    return meta, plt_props


def get_xsection(x_range, y_range):
    (x0, x1) = x_range
    t1 = time.perf_counter()
    meta, plt_props = load_viewport_datasets(x_range)

    meta_start_in_view = meta[
        (meta['time_coverage_start (UTC)']>x0)]
    meta_end_in_view = meta[
        (meta['time_coverage_end (UTC)']<x1)]

    startvlines = hv.Spikes(meta_start_in_view['time_coverage_start (UTC)']).opts(
        color='grey', spike_length=20).opts(position=-10)
    endvlines = hv.Spikes(meta_end_in_view['time_coverage_end (UTC)']).opts(
        color='grey', spike_length=20).opts(position=-10)

    data = pd.DataFrame.from_dict(
        dict(time=meta_start_in_view['time_coverage_start (UTC)'].values,
        y=5,
        text=meta_start_in_view.index.str.replace('nrt_', '')))
    ds_labels = hv.Labels(data).opts(
        fontsize=12,#plt_props['dynfontsize'],
        text_align='left')
    plotslist = []
    if len(meta_start_in_view)>0:
        plotslist.append(startvlines)
        plotslist.append(ds_labels)
    if len(meta_end_in_view)>0:
        plotslist.append(endvlines)
    if plotslist:
        return hv.Overlay(plotslist)#reduce(lambda x, y: x*y, plotslist)
    else:
        return create_None_element('Overlay')




def get_xsection_mld(x_range, y_range):
    try:
        dscopy = utils.add_dive_column(currentobject.data_in_view).compute()
    except:
        dscopy = utils.add_dive_column(currentobject.data_in_view)
    dscopy['depth'] = -dscopy['depth']
    mld = gt.physics.mixed_layer_depth(dscopy.to_xarray(), 'temperature', thresh=0.3, verbose=True, ref_depth=5)
    gtime = dscopy.reset_index().groupby(by='profile_num').mean().time
    dfmld = pd.DataFrame.from_dict(dict(time=gtime.values, mld=-mld.rolling(10, center=True).mean().values)).sort_values(by='time').dropna()
    if len(dfmld)==0:
        import pdb; pdb.set_trace();
    print(dfmld)
    mldscatter = dfmld.hvplot.line(
                    x='time',
                    y='mld',
                    color='white',
                    alpha=0.5,
                    responsive=True,
                    )
    return mldscatter


def get_xsection_raster(x_range, y_range, contour_variable=None):
    (x0, x1) = x_range
    t1 = time.perf_counter()
    print('start raster')
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist1 = []

    if plt_props['zoomed_out']:
        metakeys = [element.replace('nrt', 'delayed') for element in meta.index]
    else:
        metakeys = [element.replace('nrt', 'delayed') if
            element.replace('nrt', 'delayed') in all_datasets.index else
            element for element in meta.index]
    if contour_variable:
        variable=contour_variable
    else:
        variable=currentobject.pick_variable
    varlist = []
    for dsid in metakeys:
        ds = dsdict[dsid]
        ds = ds[ds.profile_num % plt_props['subsample_freq'] == 0]
        varlist.append(ds)

    if currentobject.pick_mld:
        varlist = utils.voto_concat_datasets(varlist)
    if varlist:
        # concat and drop_duplicates could potentially be done by pandarallel
        if currentobject.pick_TS:
            nanosecond_iterator = 1
            for ndataset in varlist:
                ndataset.index = ndataset.index + +np.timedelta64(nanosecond_iterator,'ns')
                nanosecond_iterator+=1
        dsconc = dd.concat(varlist)
        dsconc = dsconc.loc[x_range[0]:x_range[1]]
        # could be parallelized
        if currentobject.pick_TS:
            try:
                dsconc = dsconc.drop_duplicates(subset=['temperature', 'salinity']).compute()
            except:
                dsconc = dsconc.drop_duplicates(subset=['temperature', 'salinity'])
        currentobject.data_in_view = dsconc
        mplt = create_single_ds_plot_raster(data=dsconc, variable=variable)
        t2 = time.perf_counter()
        print(t2-t1)
        return mplt
    else:
        return create_None_element('Overlay')


def get_xsection_raster_contour(x_range, y_range):
    # This function exists because I cannot pass variables directly
    variable=currentobject.pick_contours
    return get_xsection_raster(x_range, y_range, contour_variable=variable)


def get_xsection_TS(x_range, y_range):
    dsconc = currentobject.data_in_view
    t1 = time.perf_counter()
    thresh = dsconc[['temperature', 'salinity']].quantile(q=[0.01, 0.99])
    t2 = time.perf_counter()
    # variable=currentobject.pick_variable
    mplt = dsconc.hvplot.scatter(
        x='salinity',
        y='temperature',
        c=currentobject.pick_variable,
        )[thresh['salinity'].iloc[0]-0.5:thresh['salinity'].iloc[1]+0.5,
          thresh['temperature'].iloc[0]-0.5:thresh['temperature'].iloc[1]+0.5]

    return mplt


def get_xsection_profiles(x_range, y_range):
    dsconc = currentobject.data_in_view
    t1 = time.perf_counter()
    thresh = dsconc[currentobject.pick_variable].quantile(q=[0.01, 0.99])
    t2 = time.perf_counter()
    # variable=currentobject.pick_variable
    mplt = dsconc.hvplot.scatter(
        x=currentobject.pick_variable,
        y='depth',
        c=currentobject.pick_variable,
        )[thresh[currentobject.pick_variable].iloc[0]:thresh[currentobject.pick_variable].iloc[1]]#,
         #thresh['temperature'].iloc[0]-0.5:thresh['temperature'].iloc[1]+0.5]

    return mplt


def get_density_contours(x_range, y_range):
    ######TSPLOT############
    # Calculate how many gridcells we need in the x and y dimensions
    import gsw

    dsconc = currentobject.data_in_view
    t1 = time.perf_counter()
    thresh = dsconc[['temperature', 'salinity']].quantile(q=[0.001, 0.999])
    #import xarray as xr
    smin,smax = (thresh['salinity'].iloc[0]-1, thresh['salinity'].iloc[1]+1)
    tmin,tmax = (thresh['temperature'].iloc[0]-1, thresh['temperature'].iloc[1]+1)

    #import pdb; pdb.set_trace();
    xdim = round((smax-smin)/0.1+1,0)
    ydim = round((tmax-tmin)+1,0)

    # Create empty grid of zeros
    dens = np.zeros((int(ydim),int(xdim)))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,int(ydim))+tmin
    si = np.linspace(1,xdim-1,int(xdim))*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.rho(si[i],ti[j],0)

    # Substract 1000 to convert to sigma-t
    dens = dens - 1000

    #da = xr.DataArray(dens, coords={'temperature': ti,'salinity': si}, dims=["temperature", "salinity"]).to_pandas()
    #da = pd.Dataset.from_dict()
    #import pdb; pdb.set_trace()

    dcont = hv.QuadMesh((si, ti, dens))
    dcont = hv.operation.contours(
        dcont,
        #overlaid=True,
        ).opts(
        show_legend=False,
        cmap='dimgray',
        )
    # this is good but the ranges are not yet automatically adjusted.
    # also, maybe the contour color should be something more discrete
    ##########END TSPLOT######
    return dcont

def create_None_element(type):
    # This is just a hack because I can't return None to dynamic maps
    if type=='Overlay':
        element = hv.Overlay(hv.HLine(0).opts(color='black', alpha=0.1)*hv.HLine(0).opts(color='black', alpha=0.1))
    elif type=='Spikes':
        element = hv.Spikes().opts(color='black', alpha=0.1)
    return element


class GliderExplorer(param.Parameterized):

    pick_variable = param.Selector(
        default='temperature', objects=[
        'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration', 'cdom', 'backscatter_scaled', 'methane_concentration'],
        label='variable', doc='Variable used to create colormesh')
    pick_basin = param.Selector(
        default='Bornholm Basin', objects=[
        'Bornholm Basin', 'Eastern Gotland',
        'Western Gotland', 'Skagerrak, Kattegat',
        'Åland Sea'], label='SAMBA observatory'
    )
    pick_cnorm = param.Selector(
        default='linear', objects=['linear', 'eq_hist', 'log'], doc='Colorbar Transformations', label='Colourbar Scale')
    pick_aggregation = param.Selector(
        default='mean', objects=['mean', 'std'], label='Data Aggregation',
        doc='Method that is applied after binning')
    pick_mld = param.Boolean(
        default=False, label='MLD', doc='Show Mixed Layer Depth')
    pick_TS = param.Boolean(
        default=False, label='Show TS-diagram', doc='Activate salinity-temperature diagram')
    pick_profiles = param.Boolean(
        default=False, label='Show profiles', doc='Activate profiles diagram')
    pick_TS_colored_by_variable = param.Boolean(
        default=False, label='Colour TS by variable', doc='Colours the TS diagram by "variable" instead of "count of datapoints"')
    pick_contours = param.Selector(default=None, objects=[
        None, 'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration', 'cdom', 'backscatter_scaled', 'methane_concentration'],
        label='contour variable', doc='Variable presented as contour')
    pick_high_resolution = param.Boolean(
        default=False, label='Increased Resolution', doc='Increases the rendering resolution (slower performance)')


    # import pdb; pdb.set_trace();

    #default=False, label='Contours', doc='add contours to the colormesh figure')
    #pick_contours_variable = param.ObjectSelector(
    #    default='potential_density', objects=[
    #    'temperature', 'salinity', 'potential_density',
    #    'chlorophyll','oxygen_concentration', 'cdom', 'backscatter_scaled', 'methane_concentration'],
    #    label='contour variable', doc='Variable presented as contour')
    # create a button that when pushed triggers 'button'
    #button_inflow = param.Action(lambda x: x.param.trigger('button_inflow'), label='Animation event example')
    data_in_view = None
    contour_processing = False
    #stream_used = False
    # on initial load, show all data
    startX, endX = (metadata['time_coverage_start (UTC)'].min().to_datetime64(),
                    metadata['time_coverage_end (UTC)'].max().to_datetime64())
    startY, endY = (None, 8)
    annotations = []
    about = """\
    # About
    This is designed to visualize data from the Voice of the Ocean SAMBA observatories. For additional datasets, visit observations.voiceoftheocean.org.
    """
    markdown = pn.pane.Markdown(about)

    def keep_zoom(self,x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    '''
    @param.depends('button_inflow', watch=True)
    def execute_event(self):
        self.markdown.object = """\
        # Baltic Inflows
        Baltic Inflows are transporting salt and oxygen into the depth of the Baltic Sea.
        """
        self.pick_variable = 'temperature'
        time.sleep(5)
        print('event:plot reloaded')
        text_annotation = hv.Text(
            x=np.datetime64('2023-12-07'),
            y=20, text='Look at this!',
            fontsize=10,
            )
        self.startX = np.datetime64('2023-12-01')
        self.endX = np.datetime64('2023-12-14')
        self.annotations.append(text_annotation)
        self.pick_variable = 'oxygen_concentration'

        return #self.dynmap*text_annotation
    '''

    @param.depends('pick_basin', watch=True)
    def change_basin(self):
        # bug: setting watch=True enables correct reset of (y-) coordinates, but leads to double initialization (slow)
        # setting watch=False fixes initialization but does not keep y-coordinate.
        x_range=(
        metadata[metadata['basin']==self.pick_basin]['time_coverage_start (UTC)'].min().to_datetime64(),
        metadata[metadata['basin']==self.pick_basin]['time_coverage_end (UTC)'].max().to_datetime64())
        self.startX, self.endX = x_range
        self.startY = None
        self.endY = 12

    #@pn.cache(max_items=2, policy='FIFO')
    @param.depends('pick_cnorm','pick_variable', 'pick_aggregation',
        'pick_mld', 'pick_basin', 'pick_TS', 'pick_contours', 'pick_TS_colored_by_variable', 'pick_high_resolution', 'pick_profiles') # outcommenting this means just depend on all, redraw always
    def create_dynmap(self):

        # import pdb; pdb.set_trace();
        #if self.pick_TS:
        #    self.pick_profiles = False
        #else:
        #    self.pick_TS_colored_by_variable.constant = True
        #if self.pick_profiles:
        #    self.pick_TS_colored_by_variable = False
        #    self.pick_TS = False

        commonheights = 500
        x_range=(self.startX,
                 self.endX)
        y_range=(self.startY,
                 self.endY)
        range_stream = RangeXY(x_range=x_range, y_range=y_range)
        range_stream.add_subscriber(self.keep_zoom)
        #self.keep_zoom(x_range, y_range)
        global currentobject
        currentobject = self
        t1 = time.perf_counter()
        pick_cnorm='linear'

        dmap_raster = hv.DynamicMap(
            get_xsection_raster,
            streams=[range_stream],
        )

        if self.pick_aggregation=='mean':
            means = dsh.mean(self.pick_variable)
        if self.pick_aggregation=='std':
            means = dsh.std(self.pick_variable)
        if self.pick_high_resolution:
            pixel_ratio=1.
        else:
            pixel_ratio=0.5
        #if self.pick_aggregation=='var':
        #    means = dsh.var(self.pick_variable)

        if self.pick_TS:
            dmap_TS = hv.DynamicMap(
                get_xsection_TS,
                streams=[range_stream],
                cache_size=1,)

            dcont = hv.DynamicMap(
                get_density_contours,
                streams=[range_stream]).opts(
                alpha=0.5,
                )
            #import pdb; pdb.set_trace();
            if not self.pick_TS_colored_by_variable:
                dmapTSr = rasterize(
                    dmap_TS,
                    pixel_ratio=pixel_ratio,
                    ).opts(
                    cnorm='eq_hist',
                    )
            else:
                #dmapTSr = spread(dmapTSr,
                #px=1, shape='circle')

                # This is the alternative version coloring the
                # TS plot by the chosen variable. Works well!
                # I should make it configurable
                dmapTSr = rasterize(
                    dmap_TS,
                    pixel_ratio=pixel_ratio,
                aggregator=means,
                    ).opts(
                    cnorm='eq_hist',
                    cmap=dictionaries.cmap_dict[self.pick_variable],#,cmap
                    clabel=self.pick_variable,
                    colorbar=True,
                    )

        if self.pick_profiles:
            dmap_profiles = hv.DynamicMap(
                get_xsection_profiles,
                streams=[range_stream],
                cache_size=1,)
            dmap_profilesr = rasterize(
                dmap_profiles,
                pixel_ratio=pixel_ratio,
                ).opts(
                cnorm='eq_hist',
                )

        dmap = hv.DynamicMap(
            get_xsection,
            streams=[range_stream],
            cache_size=1)
        #t2 = time.perf_counter()
        dmap_rasterized = rasterize(dmap_raster,
            aggregator=means,
            #x_sampling=8.64e13/48,
            y_sampling=0.2,
            pixel_ratio=pixel_ratio,
            ).opts(
            #invert_yaxis=True,
            colorbar=True,
            cmap=dictionaries.cmap_dict[self.pick_variable],#,cmap
            toolbar='above',
            tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan', 'hover'],
            default_tools=[],
            #responsive=True, # this currently breaks when activated with MLD
            #width=800,
            height=commonheights,
            cnorm=self.pick_cnorm,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",
            clabel=self.pick_variable)
        #adjoint = dmap_rasterized.hist()

        # Here it is important where the xlims are set. If set on rasterized_dmap,
        # zoom limits are kept, if applied in the end zoom limits won't work
        self.dynmap = spread(dmap_rasterized, px=1, how='source').opts(
                #invert_yaxis=True,
                ylim=(self.startY, self.endY),
                #xlim=(self.startX, self.endX),
                #ylim=(-8,None),
                #hooks=[plot_limits]
                )
        if self.pick_contours:
            if self.pick_contours == self.pick_variable:
                self.dynmap = self.dynmap * hv.operation.contours(
                    self.dynmap,
                    levels=10,
                    ).opts(
                        #cmap=dictionaries.cmap_dict[self.pick_contours],
                        line_width=2.,
                    )
            else:
                dmap_contour = hv.DynamicMap(
                    get_xsection_raster_contour,
                    streams=[range_stream],
                    )
                means_contour = dsh.mean(self.pick_contours)
                dmap_contour_rasterized = rasterize(dmap_contour,
                    aggregator=means_contour,
                    y_sampling=0.2,
                    pixel_ratio=pixel_ratio,
                    ).opts()
                #self.dynmap = self.dynmap #* hv.operation.contours(dmap_contour_rasterized, levels=5)
                self.dynmap = self.dynmap * hv.operation.contours(
                    dmap_contour_rasterized,
                    levels=10,
                    ).opts(
                        #cmap=dictionaries.cmap_dict[self.pick_contours],
                        line_width=2.,
                    )

        if self.pick_mld:
            dmap_mld = hv.DynamicMap(
                get_xsection_mld, streams=[range_stream], cache_size=1).opts(responsive=True)
            self.dynmap = (self.dynmap.opts(responsive=True) * dmap_mld.opts(responsive=True)).opts(responsive=True)
        for annotation in self.annotations:
            print('insert text annotations defined in events')
            self.dynmap = self.dynmap*annotation
        if self.pick_TS:
            linked_plots = link_selections(
                self.dynmap.opts(
                    #xlim=(self.startX, self.endX),
                    #ylim=(self.startY,self.endY),
                    responsive=True)
                + dmapTSr.opts(
                    responsive=True,
                    bgcolor='white').opts(padding=(0.05, 0.05),),
                unselected_alpha=0.3)
                    #xlim=(6,20),
                    #ylim=(-1.8, 20)),
                #selection_mode='union'
                #*dmap.opts(
                    #heigth=500,
                #    responsive=True,
                    #ylim=(-8,None))
                #)
            linked_plots.DynamicMap.II = dcont*linked_plots.DynamicMap.II
            #import pdb; pdb.set_trace()


            return linked_plots
        if self.pick_profiles:
            linked_plots = link_selections(
                self.dynmap.opts(
                    #xlim=(self.startX, self.endX),
                    #ylim=(self.startY,self.endY),
                    responsive=True)
                + dmap_profilesr.opts(
                    responsive=True,
                    bgcolor='white',).opts(padding=(0.05, 0.05))
                    #xlim=(6,20),
                    #ylim=(-1.8, 20)),
                #selection_mode='union'
                )#*dmap.opts(
                    #heigth=500,
                #    responsive=True,
                    #ylim=(-8,None))
                #)
            linked_plots.DynamicMap.II = linked_plots.DynamicMap.II
            #import pdb; pdb.set_trace()

            return linked_plots
            #return self.dynmap.opts(
            #xlim=(self.startX, self.endX),
            #ylim=(-8,None),
            #responsive=True,) + dmapTSr
        else:
            self.dynmap = self.dynmap*dmap.opts(
                    #opts.Labels(text_font_size='6pt')
                    )
            return self.dynmap.opts(
                responsive=True,
                )

class MetaExplorer(param.Parameterized):
    pick_serial = param.ObjectSelector(
        default='glider_serial', objects=[
        'glider_serial', 'optics_serial', 'altimeter_serial',
        'irradiance_serial','project', all_metadata.columns.values],
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
            height=400,
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

        for shape in fig['data']:
            shape['opacity'] = 0.7
        for i, d in enumerate(fig.data):
            d.width = (metadata.deployment_id%2+10)/12
        fig.layout.autosize = True
        fig.update_layout(height=400)
        #fig.layout.width='100%'
        #fig.layout.column.col
        return fig


def create_app_instance():
    glider_explorer=GliderExplorer()
    glider_explorer2=GliderExplorer()
    meta_explorer=MetaExplorer()
    #pp = pn.pane.Plotly(meta_explorer.create_timeline, config={'responsive': True, 'height':400})
    #pp = pn.pane(meta_explorer.create_timeline, height=100, sizing_mode='stretch_width')
    layout = pn.Column(
    pn.Row(
        pn.Column(# glider_explorer.param, # this would autogenerate all of them...
            pn.Row('Glider Dashboard'),
            pn.Param(glider_explorer,
                parameters=['pick_basin'],
                default_layout=pn.Column,
                show_name=False),
            pn.Column(
                pn.Param(glider_explorer,
                    parameters=['pick_variable'],
                    default_layout=pn.Column,
                    show_name=False),
                pn.Param(glider_explorer,
                    parameters=['pick_cnorm'],
                    #widgets={'pick_cnorm': pn.widgets.RadioButtonGroup},
                    show_name=False,
                    ),
                pn.Param(glider_explorer,
                    parameters=['pick_aggregation'],
                    #widgets={'pick_aggregation': pn.widgets.RadioButtonGroup},
                    show_name=False,
                    show_labels=True,
                    ),
                pn.Param(glider_explorer,
                    parameters=['pick_contours'],
                    show_name=False,),
                styles={'background': '#f0f0f0'},),

            #pn.Param(glider_explorer,
            #    parameters=['pick_variable'],
            #    show_name=False,),
            pn.Param(glider_explorer,
                parameters=['pick_TS', 'pick_TS_colored_by_variable'],
                default_layout=pn.Row,
                show_name=False,
                display_threshold=10,),
            pn.Param(glider_explorer,
                parameters=['pick_high_resolution'],
                show_name=False,
                display_threshold=10,),
            pn.Param(glider_explorer,
                parameters=['pick_profiles'],
                show_name=False,
                display_threshold=10,),
            #width=300,
        ),
            #, 'pick_mld']),
            #glider_explorer.pick_variable
            #),
        pn.Column(glider_explorer.create_dynmap),
        height=600,
    ),
    #pn.Row(glider_explorer2.create_dynmap),

    #pn.Row(
    #    pn.Column(glider_explorer2.param
    #        #glider_explorer.pick_basin,
    #        #glider_explorer.pick_variable
    #        ),
    #   pn.Column(glider_explorer2.create_dynmap),
    #    height=600,
    #),

    pn.Row(glider_explorer.markdown),
    pn.Row(
        pn.Column(meta_explorer.param,height=500,),
        pn.Column(meta_explorer.create_timeline,height=500,),#, sizing_mode='stretch_width'),
        height=500,
        scroll=True,#, height=420
        #sizing_mode='stretch_width'
        )
    )
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
