import xarray as xr
from pathlib import Path
from subprocess import run
import re
import regionmask
import geopandas as gpd

def clip_netcdf_to_region(global_file_path,file_path,shapefile): #This function clips netCDF to a square around regions
    gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = gdf.total_bounds
    minlon = minlon - 2.
    minlat = minlat - 2.
    maxlon = maxlon + 2.
    maxlat = maxlat + 2.
    assert minlat>0, 'Error: limited area clipping not yet tested on negative latitudes.'
    assert global_file_path.is_file(), 'Error: expected global daily .nc file not found to convert to limited area .nc file: '+global_file_path.as_posix()
    
    if minlon<0. and maxlon>=0.:
        #if region crosses the Prime Meridian, reorder data (see --msa option for multiple longitude slabs)...
        run(['ncks','-O','--msa','-d','lon,181.,360.','-d','lon,0.,180.','-d', f'lat,{minlat},{maxlat}',global_file_path.as_posix(),'processed_data/temp1.nc'])
        #...set lon>180 of global dataset to negative values...
        run(['ncap2','-O','-s',"where(lon > 180.) lon=lon-360.",'processed_data/temp1.nc','processed_data/temp2.nc'])
        #...and clip out region using the native -180,180 bounding box limits, on now-180,180-based global data 
        run(['ncks','-O','-d',f'lon,{minlon},{maxlon}','-d', f'lat,{minlat},{maxlat}','processed_data/temp2.nc',file_path.as_posix()])
    else: 
        #if not crossing the Prime Meridian, convert directly, after setting lons to 0-360.
        if minlon<0.:
            minlon = minlon + 360.
        if maxlon<0.:
            maxlon = maxlon + 360.
        run(['ncks','-O','-d',f'lon,{minlon},{maxlon}','-d', f'lat,{minlat},{maxlat}',global_file_path.as_posix(),file_path.as_posix()])
        
def mask_data_to_region(da,shapefile,region_list=None): # and this one makes and returns a mask.  Useful to separate this from above so intra-region polygons can be explored...
    print('Making regional mask...')
    #for Canada mask: incoming mask is 0-360-ranged
    #shapefile is -180,180-ranged
    
    polygons=gpd.read_file(shapefile).to_crs("EPSG:4326")
    polygons.plot()
    da[0,0,0,:,:].plot()
    geometries=[polygons.loc[n].geometry for n in range(len(polygons))]
    regions=regionmask.Regions(geometries) #apparently regionmask is smart enough to infer -180,80/0,360 longitude differences?
    area_mask=regions.mask(da.coords['lon'],da.coords['lat'])
    if region_list:
        area_mask=area_mask.where((area_mask.isin(region_list)),drop=True)
    return area_mask

def generate_nc_from_ccc(ccc_path,nc_path):
    assert ccc_path.is_file(), 'Error: expected RTD file not found to convert to missing .nc file: '+ccc_path.as_posix()
    run(['ccc2nc',ccc_path.as_posix(),nc_path.as_posix()])     

def get_file_path_and_make_files_if_needed(dir_path, runid, file_type, realm, var, region):
    if 'rtd' in file_type:
        if 'atm_lnd_ccc' in file_type:
            CanESM_suffix = "_201501_210012_rtd074.001"
            nc_file_name = "sc_"+runid+CanESM_suffix+".nc"
            file_path = dir_path / 'data' / nc_file_name
            if not file_path.is_file():
                #try to convert from ccc-formatted file so it exists to be added to matrix of file names!
                ccc_file_name = "sc_"+runid+CanESM_suffix
                ccc_file_path = dir_path / 'data' / ccc_file_name
                generate_nc_from_ccc(ccc_file_path,file_path)
        else:
            if 'phys_ocn' in file_type:
                nc_file_name = "sc_"+runid+"_201501_210012_nemo_physical_rtd.nc.001"
            elif 'bgc_ocn' in file_type:
                nc_file_name = "sc_"+runid+"_201501_210012_nemo_carbon_rtd.nc.001"
            elif 'rtd_ice_ocn' in file_type:
                nc_file_name = "sc_"+runid+"_201501_210012_nemo_ice_rtd.nc.001"
            file_path = dir_path / 'data'/ nc_file_name
    elif 'nc_monthly' in file_type:
        nc_file_name = f"{var}_{realm}_CanESM5-{runid}_esm-ssp585_r1i1p2f1_gn_201501-210012.nc"
        file_path = dir_path / "data" / "nc_output" / "CMIP6" / "CCCma" / "CCCma" / f"CanESM5-{runid}" / "esm-ssp585" / "r1i1p2f1" / f"{realm}" / f"{var}"/ "gn" / "v20190429" / nc_file_name
    elif 'nc_daily' in file_type:
        nc_file_name = f"{var}_{realm}_CanESM5-{runid}_esm-ssp585_r1i1p2f1_gn_20150101-21001231_limited_area_{region['region_name']}.nc"
        file_path = dir_path / "data" / "nc_output" / "CMIP6" / "CCCma" / "CCCma" / f"CanESM5-{runid}" / "esm-ssp585" / "r1i1p2f1" / f"{realm}" / f"{var}"/ "gn" / "v20190429" / nc_file_name               
        
        print('REDOING FILE!')
        file_path.unlink(missing_ok=True)
        if not file_path.is_file():
            #try to convert from global to limited area daily file so it exists to be added to matrix of file names!
            print('    ...converting: '+file_path.as_posix())
            global_file_name = f"{var}_{realm}_CanESM5-{runid}_esm-ssp585_r1i1p2f1_gn_20150101-21001231.nc"        
            global_file_path = dir_path / "data" / "nc_output" / "CMIP6" / "CCCma" / "CCCma" / f"CanESM5-{runid}" / "esm-ssp585" / "r1i1p2f1" / f"{realm}" / f"{var}"/ "gn" / "v20190429" / global_file_name                    
            clip_netcdf_to_region(global_file_path,file_path,region['shapefile'])
    else:
        assert False, 'Error: file type not found.'
    return file_path

def historical_ensemble_nc_filepath_matrix_maker(wc,file_type,realm=None,var=None):
    realizations=wc['realizations_to_process']
    output_dir=Path(wc['output_dir'])
    ensemble_prefix='p2-hfr'
    
    filepath_matrix = []    
    for r in realizations:
        if 'atm_lnd_ccc' in file_type:
            sd=Path(wc['CanESM_historical_atm_lnd_rtd_source_directory'])
            fname = f"sc_{ensemble_prefix}{r:02}_185001_201412_rtd074"  
            nc_fname = fname+'.nc'
            ccc_file_path = sd / fname
            file_path = output_dir / nc_fname
            if not file_path.is_file():
                generate_nc_from_ccc(ccc_file_path,file_path)      
        else:
            sd=Path(wc['CanESM_historical_ocn_rtd_source_directory'])
            if 'phys_ocn' in file_type:
                file_path = sd / f'sc_p2-hfr{r:02}_185001_201412_nemo_physical_rtd.nc'
            elif 'bgc_ocn' in file_type:
                file_path = sd / f'sc_p2-hfr{r:02}_185001_201412_nemo_carbon_rtd.nc'
            elif 'rtd_ice_ocn' in file_type:
                file_path = sd / f'sc_p2-hfr{r:02}_185001_201412_nemo_ice_rtd.nc'
        filepath_matrix.append(file_path)
    return filepath_matrix   
        
def future_ensemble_nc_filepath_matrix_maker(wc,file_type,realm=None,var=None,region=None):
    percentiles=wc['percentiles_to_process']
    realizations=wc['realizations_to_process']
    sd=Path(wc['CanESM_future_source_directory'])
    ensemble_prefixes=wc['ensemble_prefixes']
    
    filepath_matrix = []
    for p in percentiles:
        r_filepath_matrix = []
        for r in realizations:
            runid_suffix = f"p{p:02}-r{r:02}"
            #Find subensemble that contains this runid
            dir_path = None
            for ep in ensemble_prefixes:
                runid_scan = ep+"-"+runid_suffix
                dir_path_scan = sd / runid_scan 
                if dir_path_scan.is_dir():
                    dir_path = dir_path_scan
                    runid = runid_scan
                    break
            assert dir_path, 'Error: no directory found for runid: '+runid_suffix
            file_path=get_file_path_and_make_files_if_needed(dir_path,
                                                             runid,
                                                             file_type,
                                                             realm, var, region)
            r_filepath_matrix.append(file_path)
        filepath_matrix.append(r_filepath_matrix)       
    return filepath_matrix 

def clean_up_ccc_data_model(df):
    def clean_ccc_varnames(varname):
        varname=str(varname)
        pattern_list=((r'\([A-Za-z]{3} \d{4}-\d{4}\)',''),
                      (r'\s{2,}','')
                      )
        for p in pattern_list:
            varname = re.sub(p[0],p[1],varname)
        p = r'\(.*?\)'   
        units_str = re.search(p,varname)
        
        varname = re.sub(p,'',varname)
        if units_str:
            units = units_str.group(0)[1:-1]
        else:
            units = 'n/a'
        varname = varname.replace('/','p').strip().lower()
        units = units.strip().lower()
        return varname,units
    
    # In the ccc2nc version of RTD files, default time dim is weird and wrong.  Proper time is in variable 'YEAR'.
    # Replace former with latter.
    dims_to_reduce=[d for d in list(df.YEAR.dims) if d != 'ilg']
    year_values = df.YEAR.mean(dim=dims_to_reduce) #YEAR is multi-dimensional due to concatenation of many input files.  Reduce to a vector via mean, except along 'ilg' dimension which is actually time dimension
    df.coords['ilg']=year_values.values
    df = df.rename({'ilg':'year'})
    df = df.squeeze(dim='time')
    df = df.drop_vars(['YEAR','time'])
    
    # Reorganize data, so each field is it's own dataarray.  Send result to file.
    df_reorganized=xr.Dataset()
    for varname, da in df.data_vars.items():
        #get variable-specific 'type' dimension full name
        typedim=[match for match in da.dims if "type" in match][0]
        if typedim:
            for t in da[typedim]:
                da_name,da_units=clean_ccc_varnames(t.values)
                df_reorganized[da_name]=da.sel({typedim:t})
                df_reorganized[da_name].attrs['units']=da_units
        else:
            da_name,da_units=clean_ccc_varnames(t.values)
            df_reorganized[da_name]=da
            df_reorganized[da_name].attrs['units']=da_units

    # Get rid of now-redundant coordinates.
    all_coords=set(df.coords)
    coords_in_use = {dim for var in df_reorganized.data_vars for dim in df_reorganized[var].dims}
    unused_coords=all_coords-coords_in_use
    for c in unused_coords:
        try:
            df_reorganized = df_reorganized.drop_vars(c)
        except:
            print('Skipping deletion of dimension: '+str(c))

    return df_reorganized

def load_historical_ensemble_dataset(fn_matrix,wc):
    ds_array = []
    for f in fn_matrix:
        ds_array.append(xr.open_dataset(f))
    ds = xr.combine_nested(ds_array,concat_dim=['realization'],combine_attrs="override")
    ds = ds.assign_coords({'realization': wc['realizations_to_process']})
    return(ds)
    
def load_future_ensemble_dataset(fn_matrix,wc):
    ds_array = []
    for r_list in fn_matrix:
        ds_sublist = [xr.open_dataset(f) for f in r_list] #open all realization-specific datasets for a given percentile
        ds_array.append(ds_sublist)
    ds = xr.combine_nested(ds_array,concat_dim=['emission percentile', 'realization'],combine_attrs="override")
    ds = ds.assign_coords({'emission percentile': wc['percentiles_to_process'], 'realization': wc['realizations_to_process']})
    return(ds)
    
def write_ensemble_dataset(df,output_dir,fname):
        fp = output_dir / fname
        fp.unlink(missing_ok=True)    
        df.to_netcdf(fp)


        
def scp_to_collab(obj,wc):
    dest=wc['username']+'@'+wc['server']+':'+wc['collab_parent_dir']
    print(f'SCPing {obj} to: {dest}')
    run(['scp','-r',obj,dest])