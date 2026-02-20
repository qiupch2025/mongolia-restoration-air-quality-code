
laiFolderPath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/';
laiFile = fullfile(laiFolderPath, 'lai_monthly_2005_2023.nc');

lat = ncread(laiFile, 'latitude');
lon = ncread(laiFile, 'longitude');
years = 2005:2023;
n_years = length(years);

laiData = ncread(laiFile, 'lai');

s = shaperead('/home/qiupch2023/data/shp/Mongolia/Mongolia.shp');
inPoly = inpolygon(lon, lat, s.X, s.Y);


[lats, lons, ~, ~] = size(laiData);
lai_spring = nan(lats, lons, n_years);

for y = 1:n_years
    lai_spring(:, :, y) = mean(laiData(:, :, y, 3:5), 4, 'omitnan');
end


lai_trend_map = nan(lats, lons);
lai_pvalue_map = nan(lats, lons);

for r = 1:lats
    for c = 1:lons
        if inPoly(r, c)
            series = squeeze(lai_spring(r, c, :));
            if all(~isnan(series))
                X = [ones(n_years, 1), years'];
                [b, ~, ~, ~, stats] = regress(series, X);
                lai_trend_map(r, c) = b(2);
                lai_pvalue_map(r, c) = stats(3);
            end
        end
    end
end


spring_mean_series = nan(n_years, 1);
for y = 1:n_years
    temp = lai_spring(:, :, y);
    spring_mean_series(y) = mean(temp(inPoly), 'omitnan');
end
[b_mean, ~, ~, ~, stats_mean] = regress(spring_mean_series, [ones(n_years, 1), years']);
lai_mean_trend = b_mean(2);
lai_mean_pvalue = stats_mean(3);


outputFile = 'LAI_spring_trend_Mongolia.nc';

nccreate(outputFile, 'latitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'longitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'years', 'Dimensions', {'years', n_years}, 'Datatype', 'double');

nccreate(outputFile, 'lai_trend', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'lai_pvalue', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'lai_mean_trend', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'lai_mean_pvalue', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'lai_spring_mean_series', 'Dimensions', {'years', n_years}, 'Datatype', 'double');
ncwrite(outputFile, 'lai_spring_mean_series', spring_mean_series);
ncwrite(outputFile, 'latitude', lat);
ncwrite(outputFile, 'longitude', lon);
ncwrite(outputFile, 'years', years);
ncwrite(outputFile, 'lai_trend', lai_trend_map);
ncwrite(outputFile, 'lai_pvalue', lai_pvalue_map);
ncwrite(outputFile, 'lai_mean_trend', lai_mean_trend);
ncwrite(outputFile, 'lai_mean_pvalue', lai_mean_pvalue);

