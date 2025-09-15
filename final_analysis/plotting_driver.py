import yaml
from pathlib import Path

from plotting_utils import calculate_difference, calculate_da_difference, generate_dynamic_hovmeuller, generate_timeseries, flip_y_axis, get_color_ranges
import matplotlib.pyplot as plt
from processing_utils import scp_to_collab, mask_data_to_region, write_ensemble_dataset
import xarray as xr
xr.set_options(keep_attrs=True)
import numpy as np
import cmocean
import itertools
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from xclim.indices.generic import select_resample_op
from xclim.indices.stats import fit, parametric_quantile, parametric_cdf

import cftime

with open('workflow_config.yaml', "r") as conf:
    wc = yaml.load(conf, Loader=yaml.FullLoader)
output_dir=Path(wc['output_dir'])

def add_labels(ax_list):
    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for l, a in zip(labels, ax_list): #zip only pairs up to shortest length.  So, will remove unneeded labels
        a.text(-0.1, 1.05, l, transform=a.transAxes,va='top', ha='right', fontweight='bold')

# Note: following are selected in part because they 'play nice' after conversion from RTD, specifically, they don't have extra dimensions (e.g. things like soil depth, I guess)
var_list=[
    ['screen temperature global','abs_diff'],
    ['snow cover nh land','rel_diff'],
    ['ocean p-e','rel_diff'],
    ['net primary productivity','rel_diff'],
    ['runoff land','rel_diff']]

trimmed_vars=[v[0] for v in var_list]
dataset_lnd = xr.open_dataset(output_dir / "rtd_lnd_atm_ensemble_data.nc")[trimmed_vars] 

#dataset_lnd_baseline = xr.open_dataset(output_dir / "rtd_lnd_atm_ensemble_data_historical.nc")[trimmed_vars]
dataset_lnd_baseline = dataset_lnd
dataset_lnd_diff = calculate_difference(dataset_lnd, dataset_lnd_baseline, wc['historical_baseline_period'], var_list=var_list)

dataset_lnd = dataset_lnd.rename({'year':'time'})
dataset_lnd_diff = dataset_lnd_diff.rename({'year':'time'})

var_list=[['T','abs_diff'],
          ['S','abs_diff'],
          ['dp_tran','rel_diff'],
          ['atl_tran_20n','rel_diff'],
          ['mocmax20n','rel_diff'],
          ['h_tran_20N','rel_diff'],
          ['nino34','abs_diff'],
          ['nino4','abs_diff',],
          ['trop_upwell','rel_diff'],
          ['euc_max','rel_diff']]
trimmed_vars=[v[0] for v in var_list]
dataset_ocn = xr.open_dataset(output_dir / "rtd_phys_ocn_ensemble_data.nc")[trimmed_vars]
#dataset_ocn_baseline = xr.open_dataset(output_dir / "rtd_phys_ocn_ensemble_data_historical.nc")[trimmed_vars]
dataset_ocn_baseline = dataset_ocn
dataset_ocn = dataset_ocn.groupby('time.year').mean().transpose()
dataset_ocn_baseline = dataset_ocn_baseline.groupby('time.year').mean().transpose()
dataset_ocn_diff = calculate_difference(dataset_ocn, dataset_ocn_baseline, wc['historical_baseline_period'], var_list=var_list)

dataset_ocn = dataset_ocn.rename({'year':'time'})    
dataset_ocn_diff = dataset_ocn_diff.rename({'year':'time'})

var_list=[['DIC','abs_diff'],
          ['O2','abs_diff'],
          ['TAL','abs_diff']]

trimmed_vars=[v[0] for v in var_list]
dataset_ocn_bgc = xr.open_dataset(output_dir / "rtd_bgc_ocn_ensemble_data.nc")[trimmed_vars]
#dataset_ocn_baseline = xr.open_dataset(output_dir / "rtd_phys_ocn_ensemble_data_historical.nc")[trimmed_vars]
dataset_ocn_bgc_baseline = dataset_ocn_bgc
dataset_ocn_bgc = dataset_ocn_bgc.groupby('time.year').mean().transpose()
dataset_ocn_bgc_baseline = dataset_ocn_bgc_baseline.groupby('time.year').mean().transpose()
dataset_ocn_bgc_diff = calculate_difference(dataset_ocn_bgc, dataset_ocn_bgc_baseline, wc['historical_baseline_period'], var_list=var_list)

dataset_ocn_bgc = dataset_ocn_bgc.rename({'year':'time'})    
dataset_ocn_bgc_diff = dataset_ocn_bgc_diff.rename({'year':'time'})

#time_axis=[cftime_to_decimal_year(t) for t in dataset.coords['time'].values]
#dataset.coords['time']=time_axis    

var_list=[['Area_NH','rel_diff'],
          ['Area_SH','rel_diff'],
          ['Extent_NH','rel_diff'],
          ['Extent_SH','rel_diff']]
trimmed_vars=[v[0] for v in var_list]

dataset_ice = xr.open_dataset(output_dir / "rtd_ice_ocn_ensemble_data.nc")[trimmed_vars]
dataset_ice_baseline = dataset_ice
#dataset_ice_baseline = xr.open_dataset(output_dir / "rtd_ice_ocn_ensemble_data_historical.nc")[trimmed_vars]
dataset_ice_max = dataset_ice.groupby('time.year').max().transpose()
dataset_ice_min = dataset_ice.groupby('time.year').min().transpose()
dataset_ice_baseline_max = dataset_ice_baseline.groupby('time.year').max().transpose()
dataset_ice_baseline_min = dataset_ice_baseline.groupby('time.year').min().transpose()

dataset_ice_max_diff = calculate_difference(dataset_ice_max, dataset_ice_baseline_max, wc['historical_baseline_period'], var_list=var_list).rename({'year':'time'})
dataset_ice_min_diff = calculate_difference(dataset_ice_min, dataset_ice_baseline_min, wc['historical_baseline_period'], var_list=var_list).rename({'year':'time'})
dataset_ice_min = dataset_ice_min.rename({'year':'time'})
dataset_ice_max = dataset_ice_max.rename({'year':'time'})    
    
if wc['plot_global_land_hovmeuller']:    
    html_name = 'land_hovmeuller.html'
    generate_dynamic_hovmeuller(dataset_lnd_diff,html_name)
    if wc['transfer_html_plots_to_collab']:
        scp_to_collab(html_name,wc)
        
if wc['plot_global_land_timeseries']:
    html_name = 'land_timeseries.html'
    generate_timeseries(dataset_lnd_diff,html_name)
    if wc['transfer_html_plots_to_collab']:
        scp_to_collab(html_name,wc)

if wc['plot_global_ocean_and_ice_hovmeullers']:
    html_name = 'phys_ocn_hovmeuller.html'
    generate_dynamic_hovmeuller(dataset_ocn_diff,html_name)
    if wc['transfer_html_plots_to_collab']:
        scp_to_collab(html_name,wc)

    html_name = 'ice_ocn_hovmeuller.html'
    generate_dynamic_hovmeuller(dataset_ice_min_diff,html_name) #LIKELY NOT WORKING DUE TO ABOVE CHANGES
    if wc['transfer_html_plots_to_collab']:
        scp_to_collab(html_name,wc)

if wc['plot_limited_area_temperature_extreme_hovmeullers']:
    print('Making limited area extreme hovmeuller visuals...')
    variable_include_list=['heat_wave_index',
                           'heat_wave_frequency']
    dataset = xr.open_dataset(output_dir / "daily_limited_extreme_indices.nc")
    dataset = dataset[variable_include_list]
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = "weights"
    dataset_averaged = dataset.weighted(weights).mean(['lon','lat'])
    dataset_averaged = dataset_averaged.groupby('time.year').mean()
    dataset_averaged = dataset_averaged.rename({'year':'time'})
    dataset_averaged = dataset_averaged.transpose()
    for v in variable_include_list:
        dataset_averaged[v].attrs['units'] =  dataset[v].attrs['units'] #manually reassign units after loss during weighted operation
    html_name = 'extreme_hovmeuller.html'
    generate_dynamic_hovmeuller(dataset_averaged,html_name)
    if wc['transfer_html_plots_to_collab']:
        scp_to_collab(html_name,wc)

if wc['make_paper_figure_1_2_and_supps']:      
    print('Making paper figure 1: GSAT 6-panel demonstration plots...')

    coords = dataset_lnd_diff.coords
    # Generate the colors from black to red
    steps = len(coords['emission percentile'])
    colors = [(int(255 * i / (steps - 1)), 0, 0) for i in range(steps)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    fig_list = [
        {'data':dataset_lnd_diff['screen temperature global'],
         'label':'GSAT change (C)', 
         'cmap':cmocean.cm.thermal,
         'threshold':1.5,
         'file_name':'figs/Figure_1.png'},
        
        {'data':dataset_lnd_diff['snow cover nh land'],
         'label':'NH snow cover change (%)',
         'cmap':cmocean.cm.ice,
         'threshold':-33,
         'file_name':'figs/supplementary_2.png'},
        
        {'data':dataset_ocn_diff['T'],
         'label':'Ocean temperature change (C)',
         'cmap':cmocean.cm.haline,
         'threshold':0.28,
         'file_name':'figs/supplementary_3.png'},
        
        {'data':dataset_ocn_diff['nino4'],
         'label':'NINO4 SST change (C)',
         'cmap':cmocean.cm.solar,
         'threshold':1.0,
         'file_name':'figs/supplementary_4.png'},
        
        {'data':dataset_lnd_diff['net primary productivity'],
         'label':'Terrestrial NPP change (%)',
         'cmap':cmocean.cm.algae,
         'threshold':15,
         'file_name':'figs/supplementary_5.png'}
        ]

    fig_2, axes2 = plt.subplots(2,2,sharex=True,sharey=True)
    axes2_flat=axes2.flat
    fig_2.set_size_inches(9,6)
    
    for i,f in enumerate(fig_list):
        
        fig_X, ((axXa,axXb,axXc),(axXd,axXe,axXf)) = plt.subplots(2,3)
        fig_X.set_size_inches(12,6)
        
        
 #%%       Figure Xa        
        
        for nper,p in enumerate(coords['emission percentile']):
            data_per_percentile=f['data'].sel({'emission percentile':p})
            for r in coords['realization']:
               data_per_percentile.sel(realization=r).plot.line(x='time',ax=axXa,color=colors[nper],alpha=0.2,linewidth=1)
            data_per_percentile.sel(realization=coords['realization'][0]).plot.line(x='time',ax=axXa,color=colors[nper],linestyle='dashed',linewidth=1)  
        ds_median=f['data'].interp({'emission percentile':[50]}).rolling(time=6,center=True).mean().mean(dim='realization')
        ds_range=f['data'].interp({'emission percentile':[10,90]}).rolling(time=6,center=True).mean().mean(dim='realization')
        ds_median.plot.line(x='time',ax=axXa,color='blue',linestyle='solid')
        ds_range.plot.line(x='time',ax=axXa,color='blue',linestyle='dashed')
        axXa.legend().remove()
        axXa.set_xlabel('Year')
        axXa.set_ylabel(f['label'])

 #%%       Figure Xb    
        cmin, cmax=get_color_ranges(f['data'],padding=0.05)      #also used in Xc and 2
        
        im=f['data'].sel(realization=coords['realization'][0]).plot.contourf(ax=axXb,cmap=f['cmap'],levels=10,cbar_kwargs={'label':f['label'],'format':"%0.1f"},
                                                                          vmin=cmin,vmax=cmax)
        for nper,p in enumerate(coords['emission percentile']):
            axXb.hlines(p.values,min(f['data'].coords['time']),max(f['data'].coords['time']),colors=colors[nper],linestyle='dashed',linewidth=1)  
        flip_y_axis(axXb)
        axXb.set_xlabel('Year')
        axXb.set_title('')
        
 #%%    Figure Xc and Figure 2 panels
        def make_climo_hovmeuller(da, ax, cmin, cmax, thresh=None):
            climo_data=da.mean(dim='realization').rolling(time=6,center=True).mean().dropna(dim='time')
            climo_data.plot.contourf(ax=ax,cmap=f['cmap'],levels=10,vmin=cmin,vmax=cmax,cbar_kwargs={'label':f['label'],'format':"%0.1f"})
            if thresh:
                climo_data.plot.contour(levels=[f['threshold']],colors='white',linestyles='--',ax=ax)
            flip_y_axis(ax)
 
        #    Figure Xc
        make_climo_hovmeuller(f['data'], axXc, cmin, cmax, thresh=f['threshold'])       
        violin_locations=[2035,2055,2075,2095]
        axXc.hlines(50,min(f['data'].coords['time']),max(f['data'].coords['time']),colors='blue',linestyle='-')
        axXc.hlines([10,90],min(f['data'].coords['time']),max(f['data'].coords['time']),colors='blue',linestyle='--')        
        axXc.vlines(violin_locations,5,95,colors='purple',linestyle='--')
        
        #    Figure 2.  This replaces output of 1c, for non-GSAT variables
        if i>0: #skip first dataset, i.e. GSAT, .  Don't include threshold line
            make_climo_hovmeuller(f['data'], axes2_flat[i-1], cmin, cmax)

 #%%       Figure Xd
        tv = f['threshold']
        units=f['data'].attrs['units']
        c = xr.where(f['data']>tv,1,0)
        mask = c.sum(dim=['realization']).rolling(time=6,center=True).sum()/30.*100.
        mask.plot.contourf(ax=axXd,cmap=f['cmap'],levels=10,vmin=0.,vmax=100.,cbar_kwargs={'label':'% of years above threshold','format':"%0.0f"})
        mask.plot.contour(levels=[50],colors='white',linestyles='--',ax=axXd)        
        axXd.set_ylabel('Emission exceed. prob. (%)')
        flip_y_axis(axXd)

 #%%       Figure Xe        
        data_list=[]
        for cy in violin_locations:
            per_year_data=f['data'].sel(time=cy).values.flatten()
            climatological_data=f['data'].sel(time=slice(cy-5,cy+5)).values.flatten()
            data_list.append(climatological_data)
        axXe.set_xticks(np.arange(1, len(violin_locations) + 1), labels=violin_locations)
        parts = axXe.violinplot(data_list)
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cbars'].set_color('black')
        for pc in parts['bodies']:
            pc.set_facecolor('purple')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        axXe.set_ylabel(f['label'])
        
 #%%       Figure Xf
    
        # Calculate smoothed time series
        data_rolling = f['data'].rolling(time=6,center=True).mean().mean(dim='realization')
        # Calculate a detrended, non-smoothed time series
        
        emission_var = data_rolling.var(dim='emission percentile').dropna(dim='time')
        data_detrended = f['data'] - data_rolling #at each emission level, remove emission-level-specific climo trends.

        natural_var = data_detrended.var(dim=['realization','emission percentile']).rolling(time=10,center=True).mean()

        #natural_var = data_detrended.var(dim=['emission percentile']).rolling(time=6,center=True).mean().mean(dim='realization').dropna(dim='time')
        tot = emission_var+natural_var.dropna(dim='time')
        n,e =  natural_var/tot, emission_var/tot
        
        axXf.stackplot(e.time,e,n,labels=['emission','variability'])
        axXf.legend(loc='upper left')
        axXf.set_ylabel('Fraction of total variance')
    
# Finalize figure X
        [ax.autoscale(enable=True, axis='both', tight=True) for ax in [axXa,axXb,axXc,axXd,axXf]]
        [ax.set_xlabel('Year') for ax in [axXa,axXb,axXc,axXd,axXe,axXf]]
        fig_X.tight_layout()
        add_labels([axXa,axXb,axXc,axXd,axXe,axXf])
        fig_X.savefig(f['file_name'])
        if wc['transfer_paper_plots_to_collab']:
            scp_to_collab(f['file_name'],wc)        
        
 # Finalize figure 2       
    [ax.autoscale(enable=True, axis='both', tight=True) for ax in axes2_flat]
    [ax.set_xlabel('Year') for ax in axes2.flat]
    add_labels(axes2.flat)
    fname='Figure_2.png'
    fig_2.savefig(fname)
    if wc['transfer_paper_plots_to_collab']:
        scp_to_collab(fname,wc)        


# %%

if wc['make_paper_figure_supp_6']:
    from gp import experiment_4, graph_experiment_4
    rmse, meta = experiment_4(dataset_lnd_diff, varname="screen temperature global")
    fig = graph_experiment_4(rmse, meta)
    fname = 'figs/supplementary_6.png'
    fig.savefig(fname)   
    if wc['transfer_paper_plots_to_collab']:
        scp_to_collab(fname,wc)

if wc['make_paper_figures_3_4']:
    
    # TODO: request OOO & QA review of following
    # DH start primary review.
    
    # Set configs for regional extreme case studies.

    sahel_extreme_precip_fig = {'analysis_name':'extreme_precip',
             'region':'sahel',
             'shapefile':'boundary_shapefiles/Sahel.shp',
             'aep':0.02,
             'v':'pr',
             'event_duration_days':5,
             'climo_year_length':20,
             'a_label':'2% AEP extreme rain change (%)',
             'b_label':'Extreme rain AEP (%)',
             'inset_map_extent': [-20,54,-38, 41],
             'cmap':cmocean.cm.rain,
             'diff_type':'rel_diff'}    
    
    bc_extreme_heat_fig = {'analysis_name':'extreme_heat',
             'region':'bc',
             'shapefile':'boundary_shapefiles/BC.shp',
             'aep':0.02,
             'v':'tas',
             'event_duration_days':5,
             'climo_year_length':20,
             'a_label':'2% AEP extreme heat change (C)',
             'b_label':'Extreme heat AEP (%)',
             'inset_map_extent': [-141, -53, 35, 85],
             'cmap':cmocean.cm.thermal,
             'diff_type':'abs_diff'}    
        
    #for f in [bc_extreme_heat_fig]:
    for f in [sahel_extreme_precip_fig, bc_extreme_heat_fig]:
        print(f"Making paper figure: {f['analysis_name']}")
        
        # Define block stat and distribution to use for fitting
        block_statistic='max'
        from lmoments3.distr import gev
        distname="GEV"
        dist=gev
        #distname="Gumbel"
        #dist="gumbel_r"
        
        # Load square-clipped (but otherwise raw) CanESM daily data
        datafile = f"processed_data/daily_limited_{f['v']}_{f['region']}.nc"        
        da_full=xr.open_dataset(datafile)[f['v']]
        
        # Further mask data to polygon regions (set to NaN elsewhere).
        # Done here versus in original clipping so that subregions can be explored without needing full ensemble-wide re-clippingg
        if f['shapefile']:
            mask_file_name = f"processed_data/mask_{f['region']}.nc"
            if wc['redo_limited_area_masks']:
                mask = mask_data_to_region(da_full,f['shapefile'])
                mask.to_netcdf(mask_file_name)
            else:
                mask = xr.open_dataset(mask_file_name)['mask']
            da_full = da_full.where(mask.notnull())
        
        # Save some diagnostic stats
        time_realization_sample_size=f['climo_year_length']*da_full.coords['realization'].count().values
        spatial_sample_size=int(da_full.count(dim=['lat','lon']).mean().values)
        
        #Set output file name
        fname=f"{f['analysis_name']}_{f['region']}.nc"
        
        #DH start primary review.
        
        # If reprocessing requested, rerun.  Otherwise, load previous data from file.
        if wc['reprocess_spatial_aeps']:
            
            # Smooth raw daily data using rolling average with duration equal to the specified event duration
            da_smoothed=da_full.rolling(time=f['event_duration_days']).mean() # daily rolling, gridded
            
            # Calculate the annual block statistic
            block_stat_gridded = select_resample_op(da_smoothed, op=block_statistic, freq="YE") #annual block maxima across years
            
            # Set up empty data array to hold calculated AEP values.  Initially populate with NaN.
            c = block_stat_gridded.coords
            years = c['time'][:-f['climo_year_length']].values
            plevs = c['emission percentile'].values
            lats,lons = c['lat'].values, c['lon'].values
            shape = (len(years), len(plevs), len(lats), len(lons))
            aep_likelihood_gridded = xr.DataArray(np.full(shape, np.nan) ,
                                dims=["time", "emission percentile", "lat", "lon"],
                                coords={"time":years, "emission percentile":plevs, "lat":lats, "lon":lons},
                                attrs={'units':'AEP likelihood'})
            aep_magnitude_gridded = xr.DataArray(np.full(shape, np.nan) ,
                                dims=["time", "emission percentile", "lat", "lon"],
                                coords={"time":years, "emission percentile":plevs, "lat":lats, "lon":lons},
                                attrs={'units':'AEP magnitude'})
            
            # Calculate AEP results each year:
            for y in years:
                print('Processing spatially resolved block stats for: '+str(y))
                
                # Set climatological window, based on 'climo_year_length'
                ys = y
                ye = cftime.DatetimeNoLeap(ys.year + f['climo_year_length'], ys.month, ys.day)
                
                # Extract block values for climo window from full record
                block_stat = block_stat_gridded.sel({'time':slice(ys,ye)}) # annual block maxima, gridded
                
                # To take advantage of n realizations in fitting, stack time and realization data, since they are statistically equivalent.  
                # Rename this new hybrid dimension 'time' since params function hardcoded to look for 'time.
                # Dimensions: [emission percentile, lat, lon, time]
                block_stat=block_stat.rename({'time':'tmp'}).stack(time=['tmp','realization'])
                
                # Develop distribution fit params for climatological block stat along hybrid 'time'-labelled dimension
                params = fit(block_stat, dist=dist, method="PWM")
                
                # If first pass, save 'recent historical' magnitude for specified AEP.
                # This is so that future AEPs for this magnitude can be calculated (below).
                if ys==years[0]:
                    rp_mag_0 = parametric_quantile(params,q=1.-f['aep']).squeeze()
                
                aep_magnitude_gridded.loc[ys,:,:,:] =  parametric_quantile(params,q=1.-f['aep']).squeeze()   
                
                # Loop over emission percentiles, lats, and lons: 
                # TODO JF: for elegance: vectorize this, like all other steps.  Struggling to vectorize parametric_cdf call.
                for ep, lat, lon in itertools.product(plevs,lats,lons):
                    
                    subset_selector = {'lat':lat,'lon':lon,'emission percentile':ep}                   
                    # For non-nan points on spatial domain...
                    if params.sel(subset_selector).notnull().any():
                        #Calculate AEP consistent with historical magnitude 
                        aep = (1.0 - parametric_cdf(params.sel(subset_selector),
                                                    rp_mag_0.sel(subset_selector)))

                        # Save to data array after multiply by 100 for presentation as '%'
                        aep_likelihood_gridded.loc[ys,ep,lat,lon] = aep.values[0] * 100.

            # Reset data array year coordinate labels to centre of year.
            cp_d2 = int(np.ceil(f['climo_year_length']/2))
            central_year_labels = [cftime.DatetimeNoLeap(y.year + cp_d2, ys.month, ys.day) for y in years]
            aep_likelihood_gridded.coords['time'] = central_year_labels  
            aep_magnitude_gridded.coords['time'] = central_year_labels
            
            # Write to unique file.
            write_ensemble_dataset(xr.Dataset({'aep_likelihood_gridded':aep_likelihood_gridded,
                                               'aep_magnitude_gridded':aep_magnitude_gridded,
                                               'block_stat_gridded':block_stat_gridded}),
                                   Path('processed_data'),fname)
        else:
            ds = xr.open_dataset(Path('processed_data') / fname)
            aep_likelihood_gridded = ds['aep_likelihood_gridded']
            aep_magnitude_gridded = ds['aep_magnitude_gridded']
            block_stat_gridded = ds['block_stat_gridded']
            central_year_labels = block_stat_gridded.coords['time']

        # Reduce gridded AEP data to a time series per percentile
        #TODO identify and rectify reason for transpose if possible.  It's probably an ordering thing in the .loc assignments
        weights=np.cos(np.deg2rad(aep_likelihood_gridded.lat))
        aep_likelihood_ts = aep_likelihood_gridded.weighted(weights).mean(dim=('lat','lon'),skipna=True).dropna(dim='time').T
        aep_magnitude_ts = aep_magnitude_gridded.weighted(weights).mean(dim=('lat','lon'),skipna=True).dropna(dim='time').T
        
        # Reduce gridded block maxima data to a time series per percentile (no transpose needed)
        weights=np.cos(np.deg2rad(block_stat_gridded.lat))
        block_stat_full=block_stat_gridded.weighted(weights).mean(dim=('lat','lon'),skipna=True)
        
        #DH end primary review.
        
        #Plotting follows.
        
        fig = plt.figure(figsize=(12, 6))        
        axa = fig.add_subplot(1, 2, 1)
        proj = ccrs.PlateCarree()
        axa_inset = fig.add_axes([0.01,0.6,0.3,0.3], projection=proj)
        axb = fig.add_subplot(1, 2, 2)
        
        # Plot country outlines
        axa_inset.add_feature(cfeature.OCEAN,facecolor=("lightblue"))
        axa_inset.add_feature(cfeature.LAND, edgecolor='black')
        axa_inset.add_feature(cfeature.BORDERS, linestyle='-', alpha=1)
        region = gpd.read_file(f['shapefile'])
        region.plot(ax=axa_inset, facecolor='none', edgecolor='red', linewidth=3, transform=proj)
        axa_inset.set_extent(f['inset_map_extent'], crs=proj)

        newcmap = cmocean.tools.crop_by_percent(f['cmap'], 30, which='both', N=None)
        cmap = newcmap(np.linspace(0, 1, 3))
        #da = block_stat_full.rolling(time=f['climo_year_length'],center=True,).mean().mean(dim='realization').dropna('time')
        
        da_diff = calculate_da_difference(aep_magnitude_ts,
                                          aep_magnitude_ts.sel(time=aep_magnitude_ts.coords['time'].values[0]),
                                          f['diff_type'])
        da_diff_interp = da_diff.interp({'emission percentile':10},method='linear')
        da_diff_interp.plot.line(x='time',ax=axa,color=cmap[0,:],linewidth=2)
        da_diff_interp = da_diff.interp({'emission percentile':50},method='linear')
        da_diff_interp.plot.line(x='time',ax=axa,color=cmap[1,:],linewidth=2)
        da_diff_interp = da_diff.interp({'emission percentile':90},method='linear')
        da_diff_interp.plot.line(x='time',ax=axa,color=cmap[2,:],linewidth=2)
        axa.set_ylabel(f['a_label'])
        axa.set_title('')
 
        aep_likelihood_ts_interp = aep_likelihood_ts.interp({'emission percentile':10},method='linear')
        aep_likelihood_ts_interp.plot.line(x='time',ax=axb,color=cmap[0,:],linewidth=2,label='Very likely to be exceeded')
        aep_likelihood_ts_interp = aep_likelihood_ts.interp({'emission percentile':50},method='linear')
        aep_likelihood_ts_interp.plot.line(x='time',ax=axb,color=cmap[1,:],linewidth=2,label='As likely as not to be exceeded')   
        aep_likelihood_ts_interp = aep_likelihood_ts.interp({'emission percentile':90},method='linear')
        aep_likelihood_ts_interp.plot.line(x='time',ax=axb,color=cmap[2,:],linewidth=2,label='Very unlikely to be exceeded') 
        axb.legend()
        axb.set_ylabel(f['b_label'])
        axb.set_title('')
        labels = ['a)', 'b)']
        add_labels([axa,axb])
        fig.tight_layout()
        
        fname=f"figs/{f['analysis_name']}_{f['region']}.png"
        fig.savefig(fname)
        if wc['transfer_paper_plots_to_collab']:
            scp_to_collab(fname,wc)

    