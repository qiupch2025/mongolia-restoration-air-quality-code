
matFilePath = 'ndvi_landUse_mean_growthRate_m.mat';
load(matFilePath, 'growth_rate_landUse');

NDVIS = 0.05;
NDVIV = 0.90;

geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/ideal_new/geo_em.d01_ideal_new.nc';
ncid = netcdf.open(geoFilePath, 'WRITE');

fvcid = netcdf.inqVarID(ncid, 'GREENFRAC');
fvc_org = netcdf.getVar(ncid, fvcid);
targetLonGrid = ncread(geoFilePath, 'CLONG');
targetLatGrid = ncread(geoFilePath, 'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

landUseType = ncread(geoFilePath, 'LU_INDEX');

s = shaperead('/home/qiupch2023/data/shp/world/world.shp');
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

num_years = 2023 - 2005 + 1;
months_to_modify = 1:12;

all_types = 1:19;
specific_types = [5, 10, 16];
other_types = setdiff(all_types, specific_types);
other_mean_growth_rate = nanmean(growth_rate_landUse(other_types, months_to_modify), 1);
growth_rate_landUse(10, 3:5)*100
growth_rate_landUse(16, 3:5)*100
growth_rate_landUse(5, 3:5)*100
other_mean_growth_rate(3:5)*100

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i,j)
            lu_type = landUseType(i,j);
            for month = months_to_modify
                if ismember(lu_type, specific_types)
                    growth_rate = growth_rate_landUse(lu_type, month);
                else
                    growth_rate = other_mean_growth_rate(month);
                end

                initial_gvf = fvc_org(i,j,month);
                initial_ndvi = initial_gvf * (NDVIV - NDVIS) + NDVIS;

                new_ndvi = initial_ndvi * (1 + growth_rate * num_years);

                new_gvf = (new_ndvi - NDVIS) / (NDVIV - NDVIS);

                new_gvf = min(max(new_gvf, 0), 1);

                fvc_org(i,j,month) = new_gvf;
            end
        end
    end
end

netcdf.putVar(ncid, fvcid, fvc_org);

netcdf.close(ncid);
