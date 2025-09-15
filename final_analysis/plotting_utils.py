import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import cartopy
import cartopy.crs as ccrs
import holoviews as hv
from holoviews.operation import contours
hv.extension('bokeh')


def calculate_da_difference(da,da_baseline,var_type):
     with xr.set_options(keep_attrs=True):
         da = da - da_baseline
         if var_type == 'abs_diff':
             da.attrs['units'] = da.attrs['units'] + ', absolute change'
         else:
         # If a relative diff requested, then divide absolute diff by original value
             da = da / da_baseline * 100.
             da.attrs['units'] = da.attrs['units'] + ', relative change [%]'
     return da

def calculate_difference(dataset, dataset_baseline, baseline_period, var_list=None):
    baseline_ys = str(baseline_period[0])
    baseline_ye = str(baseline_period[1])

    #Calculate mean baseline climatological value.  Here, this is done across both time and realization dimensions...
    dataset_baseline = dataset_baseline.sel(year=slice(baseline_ys, baseline_ye)).mean(dim=['year','realization'])
    #Calculate differences.  Note that calculations should be broadcast.  So, each single simulation timeseries has climatological baseline subtracted from it
    for var,var_type in var_list:
        dataset[var] = calculate_da_difference(dataset[var],dataset_baseline[var],var_type)
    return dataset

def cftime_to_decimal_year(cftime_obj):
    # Get the year start and end dates
    year_start = cftime_obj.replace(month=1, day=1, hour=0, minute=0, second=0)
    year_end = cftime_obj.replace(year=cftime_obj.year + 1, month=1, day=1, hour=0, minute=0, second=0)
    
    # Calculate the total number of days in the year (account for leap years if applicable)
    days_in_year = (year_end - year_start).days
    
    # Determine the day of the year for the cftime_obj
    day_of_year = (cftime_obj - year_start).days + (cftime_obj.hour / 24) + (cftime_obj.minute / 1440) + (cftime_obj.second / 86400)
    
    # Calculate the decimal year
    decimal_year = cftime_obj.year + (day_of_year / days_in_year)
    month_of_year = cftime_obj.month
    decimal_year = cftime_obj.year + month_of_year/12
    
    return decimal_year

def generate_dynamic_hovmeuller(dataset,output_file_name):
    coords = dataset.coords
    variable_options = list(dataset.keys())
    realization_options = np.arange(1,len(coords['realization'].values)+1)
    smoothing_options = [1,6,10]
    def plot_variable(variable_name,n_realizations,smoothing):
        da = dataset[variable_name]
        # trim down to requested realizations
        realizations = coords['realization'][:n_realizations]
        da = da.sel(dict(realization=realizations))
        da = da.mean(dim='realization',keep_attrs=True)
        da = da.rolling(time=smoothing,center=True).mean()
        image = hv.Image((coords['time'], coords['emission percentile'], da.values), label='')
        units = da.attrs['units']
        return contours(image, filled=True, levels=10).opts(line_color=None,
                                                            colorbar=True,
                                                            cmap='viridis',
                                                            width=800,height=400,
                                                            xlabel='year',
                                                            ylabel='emission exceedance probability',
                                                            yticks=[(5,'95'),(25,'75'),(50,'50'),(75,'25'),(95,'5')],
                                                            title=f"{variable_name} ({units})",
                                                            border=0,
                                                            ylim=(5,95))
    dmap = hv.DynamicMap(lambda variable, n_realizations, smoothing: plot_variable(variable, n_realizations, smoothing), 
                         kdims=['Variable','N_Realizations','Smoothing_Years']).redim.values(Variable=variable_options,
                                                                                             N_Realizations=realization_options,
                                                                                             Smoothing_Years=smoothing_options)
    hv.save(dmap, output_file_name, backend='bokeh')

def generate_timeseries(dataset,output_file_name):
    coords = dataset.coords
    # Generate the colors from black to red
    steps = len(coords['emission percentile'])
    colors = [(int(255 * i / (steps - 1)), 0, 0) for i in range(steps)]
    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    plot_list=[]
    for variable in list(dataset.keys()):
        data_array = dataset[variable]        
        units = data_array.attrs['units']
        curves = []

        for nper,p in enumerate(coords['emission percentile']):
            for r in coords['realization']:
                curve = hv.Curve((coords['time'], data_array.sel({'emission percentile':p,'realization':r})), 'time', 'value').opts(width=800,height=400,
                                                                        xlabel='year',
                                                                        ylabel=variable,
                                                                        title=f"{variable} ({units})",
                                                                        border=0,
                                                                        ylim=(float(data_array.min().values),float(data_array.max().values)),
                                                                        color=normalized_colors[nper],
                                                                        alpha=0.2,
                                                                        show_legend=False)
                curves.append(curve.relabel(f'p{p}_r{r}'))
            realization_mean = data_array.sel({'emission percentile':p}).mean(dim='realization').rolling(time=11,center=True).mean()
            curve = hv.Curve((realization_mean.coords['time'],realization_mean), 'time', 'value').opts(width=800,height=400,
                                                                   xlabel='year',
                                                                   ylabel=variable,
                                                                   title=f"{variable} ({units})",
                                                                   border=0,
                                                                   xlim=(2020,2100),
                                                                   color=normalized_colors[nper],
                                                                   alpha=1,
                                                                   line_width=2,
                                                                   show_legend=False)
            curves.append(curve.relabel(f'p{p}_r{r}_mean'))
        plot_list.append(hv.Overlay(curves))
    layout = hv.Layout(plot_list).cols(1).opts(shared_axes=False)
    hv.save(layout, output_file_name, backend='bokeh')

def make_percentile_cdfs(filepaths,data):
    
    fig, ax=plt.subplots(dpi=300)
    for p in [('2020','green'),('2050','black'),('2095','red')]:
        data.sel(time=p[0]).plot(color=p[1],label=p[0]+ 's climatology')
    plt.legend()
    plt.tight_layout()
    ax.set_title('')
    plt.savefig(os.path.join(filepaths.FigureOutputDirectory,'_'.join([data.name,'percentile_cdfs.png'])))
    plt.clf()
    
def make_map(ax,data):
    proj=ccrs.Orthographic(-120, 45)
    tran=ccrs.PlateCarree()
    p=data.plot(cmap=cm.gray,
                    transform=tran,
                    subplot_kws=dict(projection=proj,facecolor="gray"),alpha=0.5, ax=ax)
    p.axes.gridlines()
    p.axes.add_feature(cartopy.feature.OCEAN)
    p.axes.add_feature(cartopy.feature.LAND)
    plt.title('')
    
    
def flip_y_axis(ax): 
    yticks=[5,25,50,75,95]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in list(reversed(yticks))])
    ax.set_ylabel('Emission exceed. prob. (%)')
    
def get_color_ranges(data,padding=0):
    if padding == 0:
        cmin=data.min().values
        cmax=data.max().values            
    else:    
        cmin=data.quantile(padding).values
        cmax=data.quantile(1.-padding).values
    return cmin,cmax