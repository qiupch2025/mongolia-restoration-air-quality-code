import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from shapely.prepared import prep
import matplotlib.patches as mpatches
import warnings
import cmaps
warnings.filterwarnings("ignore")




plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5        
plt.rcParams['xtick.major.width'] = 1.5     
plt.rcParams['ytick.major.width'] = 1.5     
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['font.weight'] = 'normal'    




input_dir = 'G:/JGRA_DATA/'  
cities = ["Beijing", "Shenyang", "Lanzhou"] 
file_suffix = "_PM10_CWT_Analysis_Result.nc"

extent = [70, 135, 25, 60]

city_coords = {
    "Beijing":  (116.40, 39.90),
    "Shenyang": (123.43, 41.80),
    "Lanzhou":  (103.83, 36.06)
}

vmin = 0
vmax = 100 




print()
try:
    
    reader_world = shpreader.Reader(r'G:/G盘/shp/world_new/world.shp')
    shp_world = list(reader_world.geometries())
    
    reader_china = shpreader.Reader(r'G:/G盘/shp/china/china.shp')
    shp_china = list(reader_china.geometries())
    print()
except Exception as e:
    print()
    shp_world = []
    shp_china = []


print()
reader = shpreader.Reader(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))
countries = list(reader.records())
mongolia_geom, china_geom = None, None
for c in countries:
    if c.attributes['NAME'] == 'Mongolia': mongolia_geom = c.geometry
    elif c.attributes['NAME'] == 'China': china_geom = c.geometry

if not mongolia_geom or not china_geom:
    print()
    exit()

mongolia_prep = prep(mongolia_geom)
china_prep = prep(china_geom)

def calculate_contribution(ds):
    lats, lons = ds.lat.values, ds.lon.values
    cwt = ds['CWT_Final'].values
    cwt = np.nan_to_num(cwt)
    sum_m, sum_c, sum_total = 0, 0, np.sum(cwt)
    rows, cols = np.where(cwt > 0)
    for r, c in zip(rows, cols):
        p = Point(lons[c], lats[r])
        val = cwt[r, c]
        if mongolia_prep.contains(p): sum_m += val
        elif china_prep.contains(p): sum_c += val
    sum_o = sum_total - sum_m - sum_c
    return sum_m, sum_c, max(0, sum_o), sum_total




def main():
    print()
    
    
    fig = plt.figure(figsize=(13, 9), dpi=300)
    
    
    cmap = cmaps.WhiteBlueGreenYellowRed
    cmap.set_bad(alpha=0) 
    
    stats_data = {"Mongolia": [], "China": [], "Others": []}
    mesh = None 
    
    
    for i, city in enumerate(cities):
        idx = i + 1 
        ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree(), aspect="auto")

        
        fpath = os.path.join(input_dir, f"{city}{file_suffix}")
        if not os.path.exists(fpath): continue
        ds = xr.open_dataset(fpath)
        
        
        m, c, o, t = calculate_contribution(ds)
        stats_data["Mongolia"].append(m/t*100)
        stats_data["China"].append(c/t*100)
        stats_data["Others"].append(o/t*100)
        
        
        
        
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        
        ax.add_feature(cfeature.LAND, facecolor='#F0F0F0', zorder=0)
        
        ax.add_feature(cfeature.OCEAN, facecolor='#B0C4DE', zorder=0)
        
        ax.add_feature(cfeature.LAKES, facecolor='#B0C4DE', edgecolor='gray', linewidth=0.5, zorder=0)

        
        
        
        
        
        
        ax.contourf(
            ds.lon, ds.lat, ds['Dust_Source_Mask'], 
            levels=[0.5, 1.5], 
            colors=['#C2B280'], 
            alpha=0.5, 
            transform=ccrs.PlateCarree(), 
            zorder=1
        )
        
        
        
        
        plot_cwt = ds['CWT_Final'].where(ds['CWT_Final'] > 0)
        mesh = plot_cwt.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            add_colorbar=False, rasterized=True, zorder=2
        )
        
        
        
        
        
        if shp_world:
            ax.add_geometries(shp_world, ccrs.PlateCarree(),
                              edgecolor='gray', facecolor='none', 
                              linewidth=1.0, zorder=3)
        
        if shp_china:
            ax.add_geometries(shp_china, ccrs.PlateCarree(),
                              edgecolor='gray', facecolor='none', 
                              linewidth=1.0, zorder=4)

        
        
        
        
        cx, cy = city_coords[city]
        ax.plot(cx, cy, marker='*', color='r', ms=22, mec='k', mew=1.5, transform=ccrs.PlateCarree(), zorder=10)
        
        
        ax.set_xticks(np.arange(70, 140, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(30, 65, 10), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        
        letter = chr(97 + i)
        ax.set_title(f"({letter}) {city}", loc='left', fontweight='bold', pad=2, fontsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")

    
    
    
    cbar_ax = fig.add_axes([0.08, 0.08, 0.4, 0.02]) 
    
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend='max')

    
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Weighted CWT Concentration ($\mu g/m^3$)', fontweight='bold',fontsize=16)
    cb.outline.set_linewidth(2) 

    
    
    
    ax_bar = fig.add_subplot(2, 2, 4)
    
    
    c_m, c_c, c_o = "#D14949", "#E39755", '#F0F0F0'
    
    x_pos = np.arange(len(cities))
    bar_width = 0.55
    
    
    p1 = ax_bar.bar(x_pos, stats_data["Mongolia"], width=bar_width, color=c_m, label='Mongolia', 
                   edgecolor='black', linewidth=2, zorder=3)
    p2 = ax_bar.bar(x_pos, stats_data["China"], width=bar_width, bottom=stats_data["Mongolia"], 
                   color=c_c, label='China', edgecolor='black', linewidth=2, zorder=3)
    
    bottom_others = [m+c for m,c in zip(stats_data["Mongolia"], stats_data["China"])]
    p3 = ax_bar.bar(x_pos, stats_data["Others"], width=bar_width, bottom=bottom_others, 
                   color=c_o, edgecolor='black', linewidth=2, label='Others', zorder=3)
    
    
    def add_labels(stats, bottom_vals, color='white'):
        for i, val in enumerate(stats):
            if val > 5:
                height = bottom_vals[i] + val/2 if bottom_vals else val/2
                ax_bar.text(i, height, f"{val:.1f}%", ha='center', va='center', 
                           color=color, fontweight='bold', fontsize=18)
    add_labels(stats_data["Mongolia"], None, 'white')
    add_labels(stats_data["China"], stats_data["Mongolia"], 'white')

    
    ax_bar.set_ylim(0, 100)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(cities, fontweight='bold', fontsize=16)
    ax_bar.tick_params(axis='x', labelsize=16)
    ax_bar.tick_params(axis='y', labelsize=16)
    ax_bar.set_ylabel('Contribution Percentage (%)', fontweight='bold', fontsize=16)
    ax_bar.set_title("(d) Source Contribution", loc='left', fontweight='bold', pad=2, fontsize=20)
    
    
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    
    ax_bar.spines['left'].set_linewidth(1.5)
    ax_bar.spines['bottom'].set_linewidth(1.5)
    
    ax_bar.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=3, 
        frameon=False, 
        fontsize=20,
        handletextpad=0.1,  
        columnspacing=0.6   
    )

    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, wspace=0.15, hspace=0.25)

    save_path = os.path.join(input_dir, "Figure3_Final_Pub_v2.png")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight') 
    plt.savefig('./Figure3_Final_Pub_v2.png', bbox_inches='tight', dpi=600)
    print()
    

if __name__ == "__main__":
    main()