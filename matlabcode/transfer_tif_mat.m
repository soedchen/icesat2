function transfer_tif_mat(InFileName,saveFilename)
[A,R] = readgeoraster(InFileName);

A = double(A);
if(length(size(A))>2)
    A = A(:,:,2);
end
lat_limit =  R.LatitudeLimits;
lon_limit =  R.LongitudeLimits;
[lan_num,lon_num] = size(A);
lat_index = lat_limit(2):-(lat_limit(2)-lat_limit(1))/(lan_num-1):lat_limit(1);
lon_index = lon_limit(1):(lon_limit(2)-lon_limit(1))/(lon_num-1):lon_limit(2);
[LON,LAT] = meshgrid(lon_index,lat_index);
% A(A>=0) = NaN;
% A = abs(A);
save(saveFilename,'LAT','LON','A','lat_index','lon_index');
end
