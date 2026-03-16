function [X_target, targetParcelNames] = define_target_locations(T, Xcentroid)
% DEFINE_TARGET_LOCATIONS AHBA parcel centroids as target locations.
% X_target = P x 3, targetParcelNames = P x 1 cell of parcel names.
%
% [X_target, targetParcelNames] = define_target_locations(T, Xcentroid)
% T = table with dataset and tissue_or_parcel; Xcentroid = N x 3 (same row order as T).

ds = T.dataset;
if iscell(ds), ds = string(ds); end
tissue = T.tissue_or_parcel;
if iscell(tissue), tissue = string(tissue); end

ahba = (ds == "AHBA");
X_ahba = Xcentroid(ahba, :);
tissue_ahba = tissue(ahba);

[parcelNames, ~, ic] = unique(tissue_ahba);
P = length(parcelNames);
X_target = zeros(P, 3);
targetParcelNames = cell(P, 1);
for p = 1:P
    mask = (ic == p);
    X_target(p, :) = mean(X_ahba(mask, :), 1);
    targetParcelNames{p} = char(parcelNames(p));
end
end
