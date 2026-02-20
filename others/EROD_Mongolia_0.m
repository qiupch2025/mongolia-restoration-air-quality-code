clc
clear


geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Mongolia_source/geo/geo_em.d01_2023_erod0.nc';

lat_grid = ncread(geoFilePath, 'CLAT');
lon_grid = ncread(geoFilePath, 'CLONG');
size(lat_grid)

s = shaperead('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/mongolia/menggu.shp');

mongolia = s;

inPoly = inpolygon(lon_grid, lat_grid, mongolia.X, mongolia.Y);
size(inPoly)
sum(inPoly,'all')

erod_data = ncread(geoFilePath, 'EROD');
size(erod_data)

for k = 1:size(erod_data,3)
    tmp = erod_data(:,:,k);
    tmp(inPoly) = 0;
    erod_data(:,:,k) = tmp;
end

size(erod_data)

ncid = netcdf.open(geoFilePath, 'WRITE');
erod_id = netcdf.inqVarID(ncid, 'EROD');
erod_org = netcdf.getVar(ncid, erod_id);
size(erod_org)

for i = 1:size(erod_org,1)
    for j = 1:size(erod_org,2)
        if inPoly(i, j)
            erod_org(i,j,:) = erod_data(i,j,:);
        end
    end
end
netcdf.putVar(ncid, erod_id, erod_org);
netcdf.close(ncid);

