function [sv_insitu,sv_model] = grid_validation(LON, LAT,A,csv_data)
 %controll points找sentinel位置
 % sv_insitu 对应 csv_data
 % sv_model 对应 A
X=csv_data(:,2);Y=csv_data(:,1); Z=csv_data(:,3);
x=LON(1,:)';y=LAT(:,1);
% A1=zeros(size(LON));
% count=ones(size(LON));
% tem_depth=zeros(size(LON,1),size(LON,2),20);
dp=zeros(size(LON));

for i=1:numel(X)
    [~,minIdx(i)]=min(abs(x-X(i)));
    [~,minIdy(i)]=min(abs(y-Y(i)));
    %     A1(minIdy(i),minIdx(i))=(A1(minIdy(i),minIdx(i))+Z(i))/count(minIdy(i),minIdx(i));
    dp(minIdy(i),minIdx(i))=Z(i);
%     tem_depth(minIdy(i),minIdx(i),count(minIdy(i),minIdx(i)))=Z(i);
%     disp([minIdy(i),minIdx(i)]);
%     disp([X(i),Y(i),x(minIdx(i)),y(minIdy(i))]);
%     disp([LON(minIdy(i),minIdx(i)),LAT(minIdy(i),minIdx(i))]);
%     count(minIdy(i),minIdx(i))=count(minIdy(i),minIdx(i))+1;
%     i
end
dp(dp==0)=nan;
A(A==0)=nan;
sv_insitu = [LAT(~isnan(dp)&~isnan(A)),LON(~isnan(dp)&~isnan(A)),dp(~isnan(dp)&~isnan(A))];  
sv_model=[LAT(~isnan(dp)&~isnan(A)),LON(~isnan(dp)&~isnan(A)),A(~isnan(dp)&~isnan(A))];
% sv_tb = [LAT(~isnan(dp)),LON(~isnan(dp)),dp(~isnan(dp))];
end
