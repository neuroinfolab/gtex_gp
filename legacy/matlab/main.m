close all
clc
clearvars -except T
if ~exist('T')
    T=readtable('gxp_samples.csv');
end

[Xcentroid, K, Xall] = parse_coordinates(T.coordinates);
dataset=categorical(T.dataset);
[a,b,c]=unique([Xcentroid grp2idx(dataset)],'rows');

G = table2array(T(:,7:end));
for i=1:max(c)
    X(i,:)=mean(G(c==i,:),1);
    i
end

allen_atlas = X(a(:,4)==1,:);[~,allen_G]=pca(allen_atlas,'NumComponents',10);
allen_coor = a(a(:,4)==1,1:3);
allen_coor(:,1)=abs(allen_coor(:,1));

gtex_atlas = X(a(:,4)==2,:);[~,gtex_G]=pca(gtex_atlas,'NumComponents',10);
gtex_coor = a(a(:,4)==2,1:3);
gtex_coor(:,1)=abs(gtex_coor(:,1));

figure
nexttile;
scatter3(allen_coor(:,1),allen_coor(:,2),allen_coor(:,3),1000,'.');
hold on
scatter3(gtex_coor(:,1),gtex_coor(:,2),gtex_coor(:,3),1000,'.');
legend('allen','gtex')
axis equal


[allen_X,allen_Y,allen_XL,allen_YL]=plsregress(allen_coor,allen_G);
[gtex_X,gtex_Y,gtex_XL,gtex_YL]=plsregress(gtex_coor,gtex_G);

nexttile
scatter3(allen_coor(:,1),allen_coor(:,2),allen_coor(:,3),1000,allen_XL(:,1),'.');axis equal
nexttile
scatter3(gtex_coor(:,1),gtex_coor(:,2),gtex_coor(:,3),1000,gtex_XL(:,1),'.');axis equal
