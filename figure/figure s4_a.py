import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.patches as patches
from netCDF4 import Dataset
import matplotlib.colors as mcolors
import cmaps

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.3  

fig = plt.figure(figsize=(10, 7), dpi=500)
date_range = [6*7-2, 6*99-2]

axe = plt.subplot(111, projection=ccrs.PlateCarree(), aspect="auto")


levels=[0,0.1,0.5,1,2, 3, 4, 5,10,15]
rgb = [
    [255,255,255],
    [77,89,168],
    [15,90,49],
    [27,163,73],
    [168,207,56],
    [253,187,18],
    [246,140,30],
    [240,79,35],
    [237,37,37],
    [237,30,35]
]
rgb = np.array(rgb) / 255.0


ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['edust'][date_range[0]:date_range[1], :, :]


contourf = plt.contourf(lon, lat, np.mean(dust, 0), levels, colors=rgb, extend='max')


plt.title('(a) Simulated Dust Emission [µg/m$^2$/s]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')


axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())


gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))


axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(top='on', right='on', which='both')


shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)


axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))


cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
cb3.ax.tick_params(labelsize=18)
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = levels
cb3.set_ticks(ticks)
cb3.set_ticklabels([x for x in ticks])





plt.subplots_adjust(left=0.10, bottom=0.01, right=0.94, top=0.93, wspace=0.2, hspace=0.1)
plt.savefig('figure_edust.png', dpi=500)
