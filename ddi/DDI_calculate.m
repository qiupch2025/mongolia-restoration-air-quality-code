
ncFile = '/home/qiupch2023/data/MODIS/data_month/ndvi_albedo_monthly_2005_2023.nc';
ndviStack = ncread(ncFile, 'ndvi');
albedoStack = ncread(ncFile, 'albedo');
lat = ncread(ncFile, 'latitude');
lon = ncread(ncFile, 'longitude');

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

inPoly = inpolygon(lon, lat, s(47).X, s(47).Y);

ndviMaxStack = NaN(size(ndviStack, 1), size(ndviStack, 2), size(ndviStack, 3));
albedoMinStack = NaN(size(albedoStack, 1), size(albedoStack, 2), size(albedoStack, 3));

for yearIdx = 1:size(ndviStack, 3)
    ndviMax = max(ndviStack(:, :, yearIdx, :), [], 4, 'omitnan');
    albedoMin = min(albedoStack(:, :, yearIdx, :), [], 4, 'omitnan');

    ndviMaxStack(:, :, yearIdx) = ndviMax .* inPoly;
    albedoMinStack(:, :, yearIdx) = albedoMin .* inPoly;
end

ndviGlobalMax = max(ndviMaxStack(:), [], 'omitnan');
ndviGlobalMin = min(ndviMaxStack(:), [], 'omitnan');
albedoGlobalMax = max(albedoMinStack(:), [], 'omitnan');
albedoGlobalMin = min(albedoMinStack(:), [], 'omitnan');


ndviNormalizedStack = (ndviMaxStack - ndviGlobalMin) / (ndviGlobalMax - ndviGlobalMin);
albedoNormalizedStack = (albedoMinStack - albedoGlobalMin) / (albedoGlobalMax - albedoGlobalMin);

[rowIdx, colIdx] = find(inPoly);
numPoints = 100;
randomIndices = randperm(length(rowIdx), numPoints);
selectedRows = rowIdx(randomIndices);
selectedCols = colIdx(randomIndices);

ndviSelectedPoints = [];
albedoSelectedPoints = [];
selectedLatLon = [];

for i = 1:numPoints
    row = selectedRows(i);
    col = selectedCols(i);

    selectedLatLon = [selectedLatLon; lat(row, col), lon(row, col)];

    ndviSelectedPoints(:, i) = squeeze(ndviNormalizedStack(row, col, :));
    albedoSelectedPoints(:, i) = squeeze(albedoNormalizedStack(row, col, :));
end


ndviAllYears = ndviSelectedPoints(:);
albedoAllYears = albedoSelectedPoints(:);

fitType = fittype('poly1');
[fitResult, gof] = fit(ndviAllYears, albedoAllYears, fitType);

figure;
scatter(ndviAllYears, albedoAllYears, 10, 'filled');
hold on;
xFit = linspace(min(ndviAllYears), max(ndviAllYears), 100);
yFit = fitResult(xFit);
plot(xFit, yFit, 'r-', 'LineWidth', 2);
xlabel('Normalized NDVI');
ylabel('Normalized Albedo');
title('Scatter Plot with Fit');
legend('Data Points', 'Fit Line');
grid on;

saveas(gcf, 'ndvi_albedo_fit.png');

slope = fitResult.p1;
k = -1 / slope;

ddiStack = [];

for yearIdx = 1:size(ndviNormalizedStack, 3)
    ndviNormalized = ndviNormalizedStack(:, :, yearIdx);
    albedoNormalized = albedoNormalizedStack(:, :, yearIdx);

    ddiIndex = k * ndviNormalized - albedoNormalized;

    ddiStack(:, :, yearIdx) = ddiIndex;
end

latlim = [min(lat(:)) max(lat(:))];
lonlim = [min(lon(:)) max(lon(:))];
R = georefcells(latlim, lonlim, size(ddiStack(:, :, 1)));

for yearIdx = 1:size(ddiStack, 3)
    data = flipud(ddiStack(:, :, yearIdx));
    tifFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', ['ddi_', num2str(2005 + yearIdx - 1), '.tif']);
    geotiffwrite(tifFilename, single(data), R);
end

trendMatrix = NaN(size(ddiStack, 1), size(ddiStack, 2));
pValueMatrix = NaN(size(ddiStack, 1), size(ddiStack, 2));

years = 1 + (1:size(ddiStack, 3)) - 1;

for row = 1:size(ddiStack, 1)
    for col = 1:size(ddiStack, 2)
        ddiValues = squeeze(ddiStack(row, col, :));
        if ~all(isnan(ddiValues))
            mdl = fitlm(years', ddiValues);
            trendMatrix(row, col) = mdl.Coefficients.Estimate(2);
            pValueMatrix(row, col) = mdl.Coefficients.pValue(2);
        end
    end
end

trendData = flipud(trendMatrix);
tifTrendFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', 'ddi_trend.tif');
geotiffwrite(tifTrendFilename, single(trendData), R);

pValueData = flipud(pValueMatrix);
tifPValueFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', 'ddi_pvalue.tif');
geotiffwrite(tifPValueFilename, single(pValueData), R);

outputNcFile = '/home/qiupch2023/data/MODIS/DDI_end/selected_points_data.nc';

nccreate(outputNcFile, 'lat', 'Dimensions', {'points', numPoints});
nccreate(outputNcFile, 'lon', 'Dimensions', {'points', numPoints});
nccreate(outputNcFile, 'years', 'Dimensions', {'years', size(ndviSelectedPoints, 1)});
nccreate(outputNcFile, 'ndvi', 'Dimensions', {'points', numPoints, 'years', size(ndviSelectedPoints, 1)});
nccreate(outputNcFile, 'albedo', 'Dimensions', {'points', numPoints, 'years', size(ndviSelectedPoints, 1)});

ncwrite(outputNcFile, 'lat', selectedLatLon(:, 1));
ncwrite(outputNcFile, 'lon', selectedLatLon(:, 2));
ncwrite(outputNcFile, 'years', years);
ncwrite(outputNcFile, 'ndvi', ndviSelectedPoints');
ncwrite(outputNcFile, 'albedo', albedoSelectedPoints');

