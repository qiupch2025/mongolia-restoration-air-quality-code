
albedoFolderPath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/';
albedoFile = fullfile(albedoFolderPath, 'ndvi_albedo_monthly_2005_2023.nc');

lat = ncread(albedoFile, 'latitude');
lon = ncread(albedoFile, 'longitude');
years = 2005:2023;
n_years = length(years);

albedoData = ncread(albedoFile, 'albedo');

s = shaperead('/home/qiupch2023/data/shp/Inner_Mongolia/Inner_Mongolia.shp');
inPoly = inpolygon(lon, lat, s.X, s.Y);


[lats, lons, ~, ~] = size(albedoData);
albedo_spring = nan(lats, lons, n_years);

for y = 1:n_years
    spring_mean = mean(albedoData(:, :, y, 3:5), 4, 'omitnan');
    albedo_spring(:, :, y) = spring_mean;
end


albedo_trend_map = nan(lats, lons);
albedo_pvalue_map = nan(lats, lons);

for r = 1:lats
    for c = 1:lons
        if inPoly(r, c)
            series = squeeze(albedo_spring(r, c, :));
            if all(~isnan(series))
                X = [ones(n_years, 1), years'];
                [b, ~, ~, ~, stats] = regress(series, X);
                albedo_trend_map(r, c) = b(2);
                albedo_pvalue_map(r, c) = stats(3);
            end
        end
    end
end


spring_mean_series = nan(n_years, 1);
for y = 1:n_years
    temp = albedo_spring(:, :, y);
    spring_mean_series(y) = mean(temp(inPoly), 'omitnan');
end
[b_mean, ~, ~, ~, stats_mean] = regress(spring_mean_series, [ones(n_years, 1), years']);
albedo_mean_trend = b_mean(2);
albedo_mean_pvalue = stats_mean(3);


outputFile = 'Albedo_spring_trend_Inner_Mongolia.nc';

nccreate(outputFile, 'latitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'longitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'years', 'Dimensions', {'years', n_years}, 'Datatype', 'double');

nccreate(outputFile, 'albedo_trend', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'albedo_pvalue', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'albedo_mean_trend', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'albedo_mean_pvalue', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'albedo_spring_mean_series', 'Dimensions', {'years', n_years}, 'Datatype', 'double');
ncwrite(outputFile, 'albedo_spring_mean_series', spring_mean_series);

ncwrite(outputFile, 'latitude', lat);
ncwrite(outputFile, 'longitude', lon);
ncwrite(outputFile, 'years', years);
ncwrite(outputFile, 'albedo_trend', albedo_trend_map);
ncwrite(outputFile, 'albedo_pvalue', albedo_pvalue_map);
ncwrite(outputFile, 'albedo_mean_trend', albedo_mean_trend);
ncwrite(outputFile, 'albedo_mean_pvalue', albedo_mean_pvalue);

