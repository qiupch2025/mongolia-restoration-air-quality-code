
albedoData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'albedo');
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'latitude');

years = 2005:2023;
num_years = length(years);
[lats, lons, ~, num_months] = size(albedoData);

albedo_detrended_2023 = nan(lats, lons, num_months);


for r = 1:lats
    for c = 1:lons
        for month = 1:num_months
            albedoSeries = squeeze(albedoData(r, c, :, month));

            valid_idx = ~isnan(albedoSeries);

            if sum(valid_idx) >= 2
                p_albedo = polyfit(years(valid_idx), albedoSeries(valid_idx), 1);

                trend_albedo_at_2023 = p_albedo(1) * (2023 - 2005);

                albedo_detrended_2023(r, c, month) = albedoData(r, c, end, month) - trend_albedo_at_2023;
            else
                albedo_detrended_2023(r, c, month) = NaN;
            end
        end
    end
end


geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2005/geo_em.d01_2005.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

albedo_mean = zeros(nx-1, ny-1, num_months);

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);


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
                sub_data = albedo_detrended_2023(:,:,month);
                sub_data = sub_data(mask);

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

