

NDVIS = 0.05;
NDVIV = 0.90;
ndviData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'ndvi_MOD13C2');
ndvi_data = squeeze(ndviData(:, :, end, :));
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'latitude');

ndvi_data(ndvi_data == -0.3) = 0;

GVF = (ndvi_data - NDVIS) ./ (NDVIV - NDVIS);

GVF(GVF < 0) = 0;
GVF(GVF > 1) = 1;


geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

fvc_mean = zeros(nx-1, ny-1, size(GVF,3));

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & (lon_highRes >= lon_min)...
                & (lon_highRes <= lon_max);
            for month = 1:size(GVF,3)
                GVF_month = GVF(:,:,month);
                sub_data = GVF_month(mask);
                fvc_mean(i,j,month) = nanmean(sub_data,'all');
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
            fvc_org(i,j,:) = fvc_mean(i,j,:);
        end
    end
end
netcdf.putVar(ncid, fvcid, fvc_org);
netcdf.close(ncid);
