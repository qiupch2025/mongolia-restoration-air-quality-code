

albedoData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'albedo');
albedo_data = squeeze(albedoData(:, :, end, :));
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'latitude');

geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

albedo_mean = zeros(nx-1, ny-1, size(albedo_data,3));

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
            for month = 1:size(albedo_data,3)
                albedo_data_month = albedo_data(:,:,month);
                sub_data = albedo_data_month(mask);
                albedo_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

ncid = netcdf.open(geoFilePath, 'WRITE');
albedo_id = netcdf.inqVarID(ncid, 'ALBEDO12M');
albedo_org = netcdf.getVar(ncid, albedo_id);

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            albedo_org(i,j,:) = albedo_mean(i,j,:)*100;
        end
    end
end

netcdf.putVar(ncid, albedo_id, albedo_org);
netcdf.close(ncid);
