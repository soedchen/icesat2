function [sv_tb,B] = grid_sentinel_all(LON, LAT,csv_data,A)  %icesat2代入sentinel
X=csv_data(:,2);Y=csv_data(:,1); Z=csv_data(:,3);
x=LON(1,:)';y=LAT(:,1);
% A1=zeros(size(LON));
count=ones(size(LON));
tem_depth=zeros(size(LON,1),size(LON,2),20);
tic
for i=1:numel(X)
    [~,minIdx(i)]=min(abs(x-X(i)));
    [~,minIdy(i)]=min(abs(y-Y(i)));
    %     A1(minIdy(i),minIdx(i))=(A1(minIdy(i),minIdx(i))+Z(i))/count(minIdy(i),minIdx(i));
    
    tem_depth(minIdy(i),minIdx(i),count(minIdy(i),minIdx(i)))=Z(i);
%     disp([minIdy(i),minIdx(i)]);
%     disp([X(i),Y(i),x(minIdx(i)),y(minIdy(i))]);
%     disp([LON(minIdy(i),minIdx(i)),LAT(minIdy(i),minIdx(i)),count(minIdy(i),minIdx(i))]);
    count(minIdy(i),minIdx(i))=count(minIdy(i),minIdx(i))+1;
    i
end
toc
tic
tem_depth(tem_depth==0)=nan;
Depth=zeros(size(LON));
for i=1:size(LON,1)
    for j=1:size(LON,2)
        Depth(i,j)=mean(tem_depth(i,j,:),'omitnan');%median
    end
    i
end
toc
sv_tb = [LAT(~isnan(Depth)),LON(~isnan(Depth)),Depth(~isnan(Depth))];  
[~,~,len] = size(A);
for ii = 1:len
    temp = A(:,:,ii);
    B(:,ii) = temp(~isnan(Depth));
end
end
