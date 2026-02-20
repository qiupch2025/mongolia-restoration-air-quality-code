
NDVIS = 0.05;
NDVIV = 0.90;

ndviData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'ndvi_MOD13C2');
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'latitude');

years = 2005:2023;
num_years = length(years);
[lats, lons, ~, num_months] = size(ndviData);

ndvi_detrended = nan(lats, lons, num_months);


for r = 1:lats
    for c = 1:lons
        for month = 1:num_months
            ndviSeries = squeeze(ndviData(r, c, :, month));

            if all(~isnan(ndviSeries))
                p_ndvi = polyfit(years, ndviSeries, 1);

                trend_ndvi_at_2023 = p_ndvi(1) * (2023 - 2005);

                ndvi_detrended(r, c, month) = ndviData(r, c, end, month) - trend_ndvi_at_2023;
            end
        end
    end
end



GVF = (ndvi_detrended - NDVIS) ./ (NDVIV - NDVIS);

GVF(GVF < 0) = 0;
GVF(GVF > 1) = 1;


geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2005/geo_em.d01_2005.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);


GVF_grid = nan(nx-1, ny-1, num_months);

for i = 1:nx-1
    i
    for j = 1:ny-1
        if inPoly(i, j)
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & ...
                   (lon_highRes >= lon_min) & (lon_highRes <= lon_max);

            for month = 1:num_months
                sub_data = GVF(:,:,month);
                sub_data = sub_data(mask);

                GVF_grid(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end


ncid = netcdf.open(geoFilePath, 'WRITE');
fvcid = netcdf.inqVarID(ncid, 'GREENFRAC');
fvc_org = netcdf.getVar(ncid, fvcid);

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            fvc_org(i,j,:) = GVF_grid(i,j,:);
        end
    end
end

netcdf.putVar(ncid, fvcid, fvc_org);
netcdf.close(ncid);

