


laiData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc','lai');
lai_data = squeeze(laiData(:, :, end, :));
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'latitude');

geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

lai_mean = zeros(nx-1, ny-1, size(lai_data,3));

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & (lon_highRes >= lon_min) & (lon_highRes <= lon_max);
            for month = 1:size(lai_data,3)
                lai_data_month = lai_data(:,:,month);
                sub_data = lai_data_month(mask);
                lai_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

lai_mean(lai_mean<0)=0;

ncid = netcdf.open(geoFilePath, 'WRITE');
laiid = netcdf.inqVarID(ncid, 'LAI12M');
lai_org = netcdf.getVar(ncid, laiid);

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            lai_org(i,j,:) = lai_mean(i,j,:);
        end
    end
end

netcdf.putVar(ncid, laiid, lai_org);
netcdf.close(ncid);
