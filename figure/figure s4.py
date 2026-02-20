import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.ticker as mticker
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde, pearsonr, linregress
import cmaps


mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2.5


fig = plt.figure(figsize=(20, 18), dpi=300)


gs = gridspec.GridSpec(3, 2, figure=fig, 
                       height_ratios=[1, 1, 1], 
                       width_ratios=[5, 4])


gs.update(wspace=0.2, hspace=0.25)


file_dust_emission = 'G:/dust_shao04_2023.nc'
file_wrf_main = 'G:/wrf_chem_dust_output_REAL.nc'
file_modis_aod = './MODIS_Dust_AOD_Corrected_Filtered.nc'
shp_world_path = 'G:/G盘/shp/world_new/world.shp'
shp_china_path = 'G:/G盘/shp/china/china.shp'
shp_val_list = [
    r'G:/G盘/shp/china_north_shp/dongbei.shp',
    r'G:/G盘/shp/china_north_shp/xibei.shp', 
    r'G:/G盘/shp/china_north_shp/huabei.shp',
    r'G:/G盘/shp/menggu/menggu.shp',
]


date_range = [6*7-2, 6*99-2]


shp_world = list(shpreader.Reader(shp_world_path).geometries())
shp_china = list(shpreader.Reader(shp_china_path).geometries())


print()
with nc.Dataset(file_dust_emission) as ds:
    lon_nc = ds.variables['lon'][:, :]
    lat_nc = ds.variables['lat'][:, :]
    
    dust_raw = ds.variables['edust'][date_range[0]:date_range[1], :, :]
    dust_mean = np.mean(dust_raw, axis=0)

axe1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(), aspect='auto')
levels_dust = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15]
rgb_dust = np.array([[255,255,255],[77,89,168],[15,90,49],[27,163,73],[168,207,56],
                     [253,187,18],[246,140,30],[240,79,35],[237,37,37],[237,30,35]])/255.0

cf1 = axe1.contourf(lon_nc, lat_nc, dust_mean, levels_dust, colors=rgb_dust, extend='max', transform=ccrs.PlateCarree())
axe1.set_title('(a) Simulated Dust Emission [µg/m$^2$/s]', loc='left', fontsize=30, pad=10)
axe1.set_extent([75, 135, 31.5, 55], crs=ccrs.PlateCarree())
axe1.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0.6)
axe1.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=5)
axe1.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=5)

axe1.set_xticks(np.arange(80, 136, 15), crs=ccrs.PlateCarree())
axe1.set_yticks(np.arange(35, 56, 10), crs=ccrs.PlateCarree())
axe1.xaxis.set_major_formatter(LongitudeFormatter())
axe1.yaxis.set_major_formatter(LatitudeFormatter())
axe1.tick_params(labelsize=25)





cbar_ax1 = fig.add_axes([0.505, 0.7, 0.009, 0.25]) 
cb1 = fig.colorbar(cf1, cax=cbar_ax1, orientation='vertical')


def fmt(x, pos):
    if x == 0: return '0' 
    return f'{x:g}' 


cb1.ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))


cb1.ax.tick_params(labelsize=25)


print()
with nc.Dataset(file_wrf_main) as ds:
    aod_wrf_mean = np.mean(ds.variables['aod'][:], axis=0)
with nc.Dataset(file_modis_aod) as ds:
    lon_modis = ds.variables['lon'][:]
    lat_modis = ds.variables['lat'][:]
    aod_mod_mean = np.nanmean(ds.variables['Dust_AOD'][:], axis=0) if ds.variables['Dust_AOD'][:].ndim==3 else ds.variables['Dust_AOD'][:]

aod_levs = np.arange(0, 0.101, 0.01)
aod_norm = mcolors.BoundaryNorm(aod_levs, ncolors=cmaps.WhiteBlueGreenYellowRed.N, extend='max')

axe3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(), aspect='auto')
axe3.contourf(lon_nc, lat_nc, aod_wrf_mean, levels=aod_levs, cmap=cmaps.WhiteBlueGreenYellowRed, norm=aod_norm, extend='max', transform=ccrs.PlateCarree())
axe3.set_title('(c) WRF-Chem AOD', loc='left', fontsize=30, pad=10)

axe4 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree(), aspect='auto')
cf4 = axe4.contourf(lon_modis, lat_modis, aod_mod_mean, levels=aod_levs, cmap=cmaps.WhiteBlueGreenYellowRed, norm=aod_norm, extend='max', transform=ccrs.PlateCarree())
axe4.set_title('(d) MODIS Dust AOD', loc='left', fontsize=30, pad=10)




cb_aod = fig.colorbar(cf4, ax=[axe3, axe4], orientation='vertical', 
                      fraction=0.03, pad=0.06, aspect=45)



formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, -2)) 
cb_aod.ax.yaxis.set_major_formatter(formatter)


cb_aod.ax.tick_params(labelsize=25)



cb_aod.ax.yaxis.get_offset_text().set_fontsize(25)  
cb_aod.ax.yaxis.get_offset_text().set_va('bottom')  
cb_aod.ax.yaxis.get_offset_text().set_ha('center')  

cb_aod.ax.yaxis.get_offset_text().set_position((1.1, 1.05))


for ax in [axe3, axe4]:
    ax.set_extent([75, 135, 31.5, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0.6)
    ax.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax.set_xticks(np.arange(80, 136, 15), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(35, 56, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter()); ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=25)




gs = gridspec.GridSpec(4, 2, figure=fig, 
                       height_ratios=[1,0.25, 1, 0.08], 
                       width_ratios=[6, 4])


gs.update(wspace=0.3, hspace=0)
print()
axe2 = fig.add_subplot(gs[0, 1])


ds_pm10 = xr.open_dataset(file_wrf_main)
pm10_sim_mean = np.mean(ds_pm10['pm10'].values, axis=0)
lon_wrf_flat = ds_pm10['lon'].values.ravel()
lat_wrf_flat = ds_pm10['lat'].values.ravel()
points_wrf = np.column_stack((lon_wrf_flat, lat_wrf_flat))

station_info = pd.read_excel('Z:/Storage(lustre)/ProjectGroup(lzu_public)/lustre_data/EST_DATA/air_quality/zhandian.xlsx')
station_lon = pd.to_numeric(station_info['lon'], errors='coerce').values
station_lat = pd.to_numeric(station_info['lat'], errors='coerce').values

all_obs_list = []
dates_pm = pd.date_range(start="2023-03-01", end="2023-05-31", freq="D")
for d_pm in dates_pm:
    f_path_pm = f'G:/G盘/蒙古荒漠化沙尘工作/aa.huifu1/airquality/站点_20230101-20231007/china_sites_{d_pm.strftime("%Y%m%d")}.csv'
    try:
        df_pm = pd.read_csv(f_path_pm, encoding='utf-8')
        val_pm = df_pm[df_pm['type'] == 'PM10'].iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
        all_obs_list.append(val_pm)
    except: continue

pm10_obs_mean = pd.concat(all_obs_list, axis=0).mean(axis=0, skipna=True).values
mask_pm = (~np.isnan(station_lon)) & (~np.isnan(pm10_obs_mean)) & (pm10_obs_mean > 0)
y_pm_sim = griddata(points_wrf, pm10_sim_mean.ravel(), 
                    np.column_stack((station_lon[mask_pm], station_lat[mask_pm])), method='linear')


x_pm = pm10_obs_mean[mask_pm][y_pm_sim > 0]
y_pm = y_pm_sim[y_pm_sim > 0]


r_pm, p_pm = pearsonr(x_pm, y_pm)
n_count = len(x_pm)


xy_pm = np.vstack([np.log10(x_pm), np.log10(y_pm)])
z_pm = gaussian_kde(xy_pm)(xy_pm)
idx_sort = z_pm.argsort()
x_plt, y_plt, z_plt = x_pm[idx_sort], y_pm[idx_sort], z_pm[idx_sort]


sc2 = axe2.scatter(x_plt, y_plt, c=z_plt, cmap=cmaps.WhiteBlueGreenYellowRed, 
                   s=40, alpha=0.7, edgecolors='none', zorder=3)


axe2.set_xscale('log')
axe2.set_yscale('log')
axe2.set_xlim(1, 1000)
axe2.set_ylim(1, 1000)


axe2.plot([1, 1000], [1, 1000], 'k-', lw=1.5, zorder=4)          
axe2.plot([1, 1000], [5, 5000], "#3B3B3B", lw=1, ls='--', zorder=2) 
axe2.plot([1, 1000], [0.2, 200], "#3B3B3B", lw=1, ls='--', zorder=2) 


p_label = "P < 0.01" if p_pm < 0.01 else f"P = {p_pm:.2f}"
stats_text = f"N = {n_count}\nR = {r_pm:.2f}\n{p_label}"
axe2.text(0.05, 0.95, stats_text, transform=axe2.transAxes, fontsize=30, 
          fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 设置标题和标签
axe2.set_title('(b) PM$_{10}$ Performance', loc='left', fontsize=30, pad=10)
axe2.set_xlabel('Observed PM$_{10}$ [µg/m$^3$]', fontsize=25,fontweight='bold')
axe2.set_ylabel('Simulated PM$_{10}$ [µg/m$^3$]', fontsize=25,fontweight='bold')

# 设置刻度字体大小
axe2.tick_params(axis='both', which='major', labelsize=25)
axe2.tick_params(axis='both', which='minor', labelsize=14)

# 添加颜色条
cbar2 = fig.colorbar(sc2, ax=axe2, pad=0.03, aspect=20)
cbar2.ax.tick_params(labelsize=22)

print()
# ================= 5. (e) Dust AOD Validation (gs[1, 1]) =================
print()

# --- 5.1 数据对齐与裁剪 (确保 xa, ya 被定义) ---
# 准备 MODIS 的 2D 网格用于插值
lons_mod_2d, lats_mod_2d = np.meshgrid(lon_modis, lat_modis) if lon_modis.ndim==1 else (lon_modis, lat_modis)

# 将 MODIS 数据插值到 WRF 网格 (lon_nc, lat_nc)
print()
aod_mod_interp = griddata(
    (lons_mod_2d[~np.isnan(aod_mod_mean)], lats_mod_2d[~np.isnan(aod_mod_mean)]), 
    aod_mod_mean[~np.isnan(aod_mod_mean)], 
    (lon_nc, lat_nc), 
    method='linear'
)

# SHP 掩膜提取函数
def get_shp_mask(lon2d, lat2d, shp_paths):
    mask = np.zeros(lon2d.shape, dtype=bool)
    pts = np.vstack((lon2d.flatten(), lat2d.flatten())).T
    for s_path in shp_paths:
        try:
            reader = shpreader.Reader(s_path)
            for geom in reader.geometries():
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for poly in polys:
                    mask |= mpath.Path(np.array(poly.exterior.coords)).contains_points(pts).reshape(lon2d.shape)
        except: continue
    return mask

# 执行掩膜过滤，得到 xa (MODIS) 和 ya (WRF)
aod_mask = get_shp_mask(lon_nc, lat_nc, shp_val_list)
# 过滤：必须在 SHP 区域内，且双方数值均大于 1e-4 (为了取对数)
valid_idx = aod_mask & (aod_wrf_mean > 1e-4) & (aod_mod_interp > 1e-4)
xa, ya = aod_mod_interp[valid_idx], aod_wrf_mean[valid_idx]

# --- 5.2 绘图逻辑 ---
if len(xa) > 0:
    axe5 = fig.add_subplot(gs[2, 1])

    # 统计计算
    log_xa, log_ya = np.log10(xa), np.log10(ya)
    slope_a, intercept_a, r_a, p_a, _ = linregress(log_xa, log_ya)
    n_count_a = len(xa)

    # 密度计算与采样排序
    sample_num = min(len(xa), 6000)
    xa_sample, ya_sample = xa[:sample_num], ya[:sample_num]
    xy_a = np.vstack([np.log10(xa_sample), np.log10(ya_sample)])
    za = gaussian_kde(xy_a)(xy_a)
    idx_sort_a = za.argsort()
    xa_plt, ya_plt, za_plt = xa_sample[idx_sort_a], ya_sample[idx_sort_a], za[idx_sort_a]

    # 散点绘制
    sc5 = axe5.scatter(xa_plt, ya_plt, c=za_plt, cmap=cmaps.WhiteBlueGreenYellowRed, 
                       s=40, alpha=0.7, edgecolors='none', zorder=3)

    # 坐标轴设置
    axe5.set_xscale('log')
    axe5.set_yscale('log')
    # axe5.set_aspect('equal')
    axe5.set_xlim(0.01, 1)
    axe5.set_ylim(0.01, 1)

    # 绘制辅助线 (1:1, 1:2, 2:1)
    axe5.plot([0.01, 1], [0.01, 1], 'k-', lw=1.5, zorder=4)
    axe5.plot([0.01, 1], [0.02, 2], '#3B3B3B', lw=1, ls='--', zorder=2)   # 2:1
    axe5.plot([0.01, 1], [0.005, 0.5], '#3B3B3B', lw=1, ls='--', zorder=2) # 1:2

    # 标注统计信息
    p_label_a = "P < 0.01" if p_a < 0.01 else f"P = {p_a:.2f}"
    stats_text_a = f"N = {n_count_a}\nR = {r_a:.2f}\n{p_label_a}"
    axe5.text(0.05, 0.95, stats_text_a, transform=axe5.transAxes, fontsize=30, 
              fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 标题与标签
    axe5.set_title('(e) Dust AOD Validation', loc='left', fontsize=30, pad=10)
    axe5.set_xlabel('MODIS AOD', fontsize=25,fontweight='bold')
    axe5.set_ylabel('WRF AOD', fontsize=25,fontweight='bold')
    axe5.tick_params(axis='both', which='major', labelsize=25)

    # 颜色条
    cbar5 = fig.colorbar(sc5, ax=axe5, pad=0.03, aspect=20)
    cbar5.ax.tick_params(labelsize=22)
else:
    print()

print()


# ==================================================================
# Final Global Tick Styling
# ==================================================================
for ax in fig.get_axes():
    # Set major ticks (thickness and length)
    ax.tick_params(axis='both', which='major', direction='out', 
                   width=2.5, length=8, labelsize=25)
    
    # Set minor ticks (thickness and length)
    ax.tick_params(axis='both', which='minor', direction='out', 
                   width=2.0, length=4)
    
    # Ensure spines (box edges) match the thickness
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

# Special handling for Cartopy gridliner labels if they were drawn separately
# (This ensures geo-axes labels stay consistent with your specific requirements)

# ================= 6. 保存 =================
plt.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.06)
plt.savefig('Final_Corrected_AOD_PM10_Figure.png', dpi=600)
print()