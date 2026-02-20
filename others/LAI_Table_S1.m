
laiFile = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Inner_Mongolia/lai_monthly_max_2001_2023_Inner_Mongolia.nc';
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
shpPath = '/home/qiupch2023/data/shp/Inner_Mongolia/Inner_Mongolia.shp';
saveFileName = 'lai_landUse_mean_growthRate.mat';

lat_highRes = ncread(laiFile, 'latitude');
lon_highRes = ncread(laiFile, 'longitude');
laiData = ncread(laiFile, 'lai_max');
laiData = laiData(:,:,5:23,:);
targetLatGrid = ncread(geoFilePath, 'CLAT');
targetLonGrid = ncread(geoFilePath, 'CLONG');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);
years = 2005:2023;
n_years = length(years);
months = 12;
s = shaperead(shpPath);
inPoly = inpolygon(targetLonGrid, targetLatGrid, s.X, s.Y);

lai_target = nan(nx-1, ny-1, n_years, months);

for i = 1:nx-1

    for j = 1:ny-1
        if inPoly(i,j)
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            mask_highRes = (lat_highRes >= lat_min & lat_highRes <= lat_max) & ...
                           (lon_highRes >= lon_min & lon_highRes <= lon_max);

            if nnz(mask_highRes) > 0
                for y = 1:n_years
                    for m = 1:months
                        data_sub = laiData(:,:,y,m);
                        lai_target(i,j,y,m) = nanmean(data_sub(mask_highRes));
                    end
                end
            end
        end
    end
end


landUseType = ncread(geoFilePath, 'LU_INDEX');
mean_lai_byLandUse = nan(19, n_years, months);

for lu = 1:19
    lu_mask = (landUseType == lu) & inPoly;
    for y = 1:n_years
        for m = 1:months
            current_data = lai_target(:,:,y,m);
            mean_lai_byLandUse(lu,y,m) = nanmean(current_data(lu_mask));
        end
    end
end


growth_rate_landUse = nan(19, months);

for lu = 1:19
    for m = 1:months
        laiSeries = squeeze(mean_lai_byLandUse(lu,:,m))';
        if all(~isnan(laiSeries))
            X = [ones(n_years,1), years'];
            b = regress(laiSeries, X);
            trend = b(2);
            fitted_2005 = b(1) + b(2)*2005;
            if abs(fitted_2005) > eps
                growth_rate_landUse(lu,m) = trend / fitted_2005;
            else
                growth_rate_landUse(lu,m) = NaN;
            end
        end
    end
end


save(saveFileName, 'mean_lai_byLandUse', 'growth_rate_landUse');
