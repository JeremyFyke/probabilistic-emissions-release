import yaml
from processing_utils import historical_ensemble_nc_filepath_matrix_maker, future_ensemble_nc_filepath_matrix_maker
from processing_utils import load_historical_ensemble_dataset, load_future_ensemble_dataset, clean_up_ccc_data_model
from processing_utils import write_ensemble_dataset, scp_to_collab
from pathlib import Path
import xclim as xc
import xarray as xr
# %%

with open('workflow_config.yaml', "r") as conf:
    wc = yaml.load(conf, Loader=yaml.FullLoader)
output_dir=Path(wc['output_dir'])

# This command mounts the server files locally. Not useful if the data is already on disk.
# httpdirfs -f --cache https://hpfx.collab.science.gc.ca/~jef000/probabilistic-emissions/ ./cccma

#1 Remake all analysis-relevant metrics, from raw (nc-processed) CanESM RTD output

if wc['reload_rtd']:
    print('Loading rtd data for atm_lnd:')
    print('  ...historical...')
    fnames = historical_ensemble_nc_filepath_matrix_maker(wc,'rtd_atm_lnd_ccc')
    df = load_historical_ensemble_dataset(fnames,wc)
    df = clean_up_ccc_data_model(df)
    write_ensemble_dataset(df,output_dir,'rtd_lnd_atm_ensemble_data_historical.nc')
    print('  ...future...')
    fnames = future_ensemble_nc_filepath_matrix_maker(wc,'rtd_atm_lnd_ccc')
    df = load_future_ensemble_dataset(fnames,wc)
    df = clean_up_ccc_data_model(df)
    write_ensemble_dataset(df,output_dir,'rtd_lnd_atm_ensemble_data.nc')

    for ocn_rtd in ['rtd_phys_ocn', 'rtd_bgc_ocn', 'rtd_ice_ocn']:
        print(f'Loading rtd data for {ocn_rtd}')
        print('  ...historical...')        
        fnames = historical_ensemble_nc_filepath_matrix_maker(wc,ocn_rtd)
        df = load_historical_ensemble_dataset(fnames,wc)
        write_ensemble_dataset(df,output_dir,f'{ocn_rtd}_ensemble_data_historical.nc')

        print('  ...future...')        
        fnames = future_ensemble_nc_filepath_matrix_maker(wc,ocn_rtd)
        df = load_future_ensemble_dataset(fnames,wc)
        write_ensemble_dataset(df,output_dir,f'{ocn_rtd}_ensemble_data.nc')

# Global monthly data
if wc['reload_global_monthly']:
    for realm,var in (('Amon','tas'),
                      ('Amon','tasmax'),
                      ('Amon','tasmin'),
                      ('Amon','pr'),
                      ('Omon','tos')):
        print(f'Loading global data for {realm}, {var}')
        fnames = future_ensemble_nc_filepath_matrix_maker(wc,'nc_monthly',realm=realm,var=var)
        df = load_future_ensemble_dataset(fnames,wc)
        write_ensemble_dataset(df,output_dir,f'monthly_global_{var}.nc')
    
if wc['reload_limited_area_daily']:

    print('  ...historical...')
    print('NOTE: CURRENTLY YOU NEED TO RUN concatenate_hist_data_script.sh TO GET HISTORICAL DAILY using pre-2014 data if that is your jam!')

    for realm,var in (('day','tas'),
                      ('day','pr'),
                      #('day','tasmin'),
                      #('day','tasmax'),
                      #('day','hurs')
                      ):
        for region in [{'region_name':'bc','shapefile':'boundary_shapefiles/BC.shp'},
                       {'region_name':'sahel','shapefile':'boundary_shapefiles/Sahel.shp'},]:        
            print(f"Loading and concatenating limited-area daily ensemble data for {realm}, {var}, {region['region_name']}")
            fnames = future_ensemble_nc_filepath_matrix_maker(wc,'nc_daily',realm=realm,var=var, region=region)
            df = load_future_ensemble_dataset(fnames,wc)
            write_ensemble_dataset(df,output_dir,f"daily_limited_{var}_{region['region_name']}.nc")

if wc['recalculate_xclim_indices']:
    ds_hist = xr.Dataset()
    ds_fut = xr.Dataset()
    ds_fut['tasmax'] = xr.open_dataset(output_dir / "daily_limited_tasmax.nc")['tasmax'] - 273.15
    ds_fut['tasmin'] = xr.open_dataset(output_dir / "daily_limited_tasmin.nc")['tasmin'] - 273.15
    ds_fut['tasmax'].attrs['units']='degC'
    ds_fut['tasmin'].attrs['units']='degC'
    
    ds_hist['tasmax'] = xr.open_dataset(output_dir / "daily_limited_tasmax_19840101-20141231.nc")['tasmax'] - 273.15
    ds_hist['tasmin'] = xr.open_dataset(output_dir / "daily_limited_tasmin_19840101-20141231.nc")['tasmin'] - 273.15
    ds_hist['tasmax'].attrs['units']='degC'
    ds_hist['tasmin'].attrs['units']='degC'
    
    indice_ds = xr.Dataset()
    

    historical_climo=slice("1985-01-01","2014-12-31")
    q=364./365.
    tasmin_thresh=ds_hist['tasmin'].sel(dict(time=historical_climo)).quantile(q,dim=['time','realization'],skipna=True)
    tasmax_thresh=ds_hist['tasmax'].sel(dict(time=historical_climo)).quantile(q,dim=['time','realization'],skipna=True)

    full_list=[]   
    for lat in tasmax_thresh.coords['lat']:
        print('lat:',str(lat.values))
        tmp_list=[]
        for lon in tasmax_thresh.coords['lon']:
            ds=xr.Dataset()
            tasmin_vec=ds_fut['tasmax'].sel(lat=lat,lon=lon)
            tasmax_vec=ds_fut['tasmax'].sel(lat=lat,lon=lon)
            tasmin_thresh_scalar=tasmin_thresh.sel(dict(lat=lat,lon=lon)).values            
            tasmax_thresh_scalar=tasmax_thresh.sel(dict(lat=lat,lon=lon)).values
            ds['heat_wave_index']=xc.indices.heat_wave_index(tasmax_vec,
                                           thresh=f'{tasmax_thresh_scalar} degC',
                                           window=3,
                                           freq='YS')
            ds['heat_wave_index'].attrs['units']='day'
            ds['heat_wave_frequency'] = xc.indices.heat_wave_frequency(tasmin_vec, tasmax_vec,
                                             thresh_tasmin=f'{tasmin_thresh_scalar} degC',
                                             thresh_tasmax=f'{tasmax_thresh_scalar} degC',
                                             window=3,
                                             freq='YS')
            ds['heat_wave_frequency'].attrs['units']='events per year'
            tmp_list.append(ds)
        full_list.append(tmp_list)
    indice_ds = xr.combine_nested(full_list, concat_dim=['lat', 'lon'],combine_attrs="override")
    write_ensemble_dataset(indice_ds,output_dir,'daily_limited_extreme_indices.nc')

if wc['transfer_to_collab']:
    scp_to_collab(output_dir.as_posix(),wc)

    
    
