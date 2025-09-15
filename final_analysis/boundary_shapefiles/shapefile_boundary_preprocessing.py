import regionmask
import geopandas as gpd
from shapely.geometry import box
from subprocess import run
import xarray as xr


download_shapefiles=True

datasets=['ne_50m_admin_0_countries.zip','ne_50m_admin_1_states_provinces.zip']

world = gpd.read_file('ne_50m_admin_0_countries.zip')
canada = world[world["NAME"] == "Canada"]
bb = canada.total_bounds

clip_poly = box(bb[0],bb[1],bb[3],62.)  # bounds: minx, miny, maxx, maxy
clip_gdf = gpd.GeoDataFrame(geometry=[clip_poly], crs=canada.crs)
canada_south = gpd.clip(canada, clip_gdf)
canada_south.plot(edgecolor='k')
canada_south.to_file("Canada_south.shp", driver="ESRI Shapefile")

sahel = world[world['NAME'].isin(['Senegal', 'Guinea', 'Mali', 'Burkina Faso', 'Niger', 'Nigeria', 'Chad'])]
sahel = sahel.dissolve()
sahel.plot(edgecolor='k')
sahel.to_file("Sahel.shp", driver="ESRI Shapefile")

provinces = gpd.read_file('ne_50m_admin_1_states_provinces.zip')
canadian_provinces = provinces[provinces['admin'] == 'Canada']

ab_bc = canadian_provinces[canadian_provinces['postal'].isin(['AB', 'BC'])]
ab_bc_joined = ab_bc.dissolve()
ab_bc_joined.plot(edgecolor='k')
ab_bc_joined.to_file("AB_BC_joined.shp", driver="ESRI Shapefile")

bc = canadian_provinces[canadian_provinces['postal'].isin(['BC'])]
bc.plot(edgecolor='k')
bc.to_file("BC.shp", driver="ESRI Shapefile")

qc = canadian_provinces[canadian_provinces['postal'].isin(['QC'])]
qc.plot(edgecolor='k')
qc.to_file("QC.shp", driver="ESRI Shapefile")

# %% test 
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(projection=proj))
 
# Plot country outlines
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
world.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, transform=proj) 

# Plot your geometry, e.g., in red
sahel.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3, transform=proj)
plt.title('')
plt.show()
