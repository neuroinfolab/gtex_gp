% Load and format data: load CSV, parse coords, match HVG genes, define parcels, assign parcel per row.
% Y_subj_parcel_gene uses HVG; atlases use ALL genes. Atlas = one row per unique parcel, mean over all samples at that parcel.
% Requires: load_gxp_data, parse_coordinates, match_gene_names, define_target_locations
function out = load_and_format()
root = fileparts(mfilename('fullpath'));

%% Load CSV and schema checks
T = load_gxp_data(fullfile(root, 'gxp_samples.csv'));
assert(height(T) >= 1);
assert(all(ismember({'subject','age','sex','dataset','tissue_or_parcel','coordinates'}, T.Properties.VariableNames)));
ds = string(T.dataset);
assert(sum(ds == "GTEx") >= 1 && sum(ds == "AHBA") >= 1);
coord = T.coordinates; if iscell(coord), coord = string(coord); end

%% Parse coordinates
coordCell = cellfun(@(c) char(c), cellstr(T.coordinates), 'Uni', false);
[Xcentroid, ~, ~] = parse_coordinates(coordCell);
assert(~any(any(isnan(Xcentroid), 2)));

%% HVG genes
fid = fopen(fullfile(root, 'ahba_100hvg.txt'), 'r');
hvg = {};
while true
    L = fgetl(fid);
    if ~ischar(L), break; end
    L = strtrim(L);
    if ~isempty(L), hvg{end+1} = L; end
end
fclose(fid);
geneVarNames = T.Properties.VariableNames;
[idxInGeneCols, matchedNames] = match_gene_names(geneVarNames, hvg);
geneColIdx = 6 + idxInGeneCols;
G = length(geneColIdx);

%% Target parcels (AHBA centroids)
[X_target, targetParcelNames] = define_target_locations(T, Xcentroid);
R = size(X_target, 1);

%% Parcel assignment per row: AHBA by name, GTEx by nearest centroid
parcel_index = nan(height(T), 1);
parcel2idx = containers.Map();
for r = 1:R
    parcel2idx(targetParcelNames{r}) = r;
end
tissue = string(T.tissue_or_parcel);
for i = 1:height(T)
    if ds(i) == "AHBA"
        tname = char(tissue(i));
        if parcel2idx.isKey(tname)
            parcel_index(i) = parcel2idx(tname);
        end
    else
        % GTEx: nearest centroid
        d = sum((X_target - Xcentroid(i,:)).^2, 2);
        [~, parcel_index(i)] = min(d);
    end
end

%% Subjects x parcels x genes: one R×G matrix per subject (AHBA + GTEx), NaN where unobserved
subj = string(T.subject);
[subjList, ~, subjIc] = unique(subj, 'stable');
nSubj = length(subjList);
Y_subj_parcel_gene = nan(nSubj, R, G);
for i = 1:nSubj
    rows = find(subjIc == i);
    Y_i = nan(R, G);
    cnt = zeros(R, G);
    for k = 1:length(rows)
        row = rows(k);
        r = parcel_index(row);
        if isnan(r) || r < 1 || r > R, continue; end
        for g = 1:G
            v = T.(geneVarNames{geneColIdx(g)})(row);
            if isnumeric(v), v = double(v); else, v = str2double(v); end
            if ~isnan(v)
                if isnan(Y_i(r,g)), Y_i(r,g) = v; cnt(r,g) = 1; else, Y_i(r,g) = Y_i(r,g) + v; cnt(r,g) = cnt(r,g) + 1; end
            end
        end
    end
    Y_i(cnt > 0) = Y_i(cnt > 0) ./ cnt(cnt > 0);
    Y_subj_parcel_gene(i, :, :) = Y_i;
end

%% Atlases: ALL genes (columns 7:end). Loop over parcels; for each parcel, average all samples at that parcel.
ahba = (ds == "AHBA");
gtex = (ds == "GTEx");
geneColIdx_all = (7:size(T, 2))';
G_all = length(geneColIdx_all);
matchedNames_all = geneVarNames(7:end);

%% Unique parcels for AHBA and GTEx
u_ahba = unique(parcel_index(ahba));
u_ahba = u_ahba(~isnan(u_ahba) & u_ahba >= 1 & u_ahba <= R);
u_ahba = sort(u_ahba);
R_ahba = length(u_ahba);
u_gtex = unique(parcel_index(gtex));
u_gtex = u_gtex(~isnan(u_gtex) & u_gtex >= 1 & u_gtex <= R);
u_gtex = sort(u_gtex);
R_gtex = length(u_gtex);

%% AHBA atlas: one row per unique parcel; row = mean of all AHBA samples at that parcel (all genes)
Y_ahba_atlas = nan(R_ahba, G_all);
xyz_ahba = zeros(R_ahba, 3);
for i = 1:R_ahba
    r = u_ahba(i);
    xyz_ahba(i, :) = X_target(r, :);
    rows = find(ahba & parcel_index == r);
    M = zeros(length(rows), G_all);
    for j = 1:G_all
        col = geneColIdx_all(j);
        v = T.(geneVarNames{col})(rows);
        if isnumeric(v), v = double(v); else, v = str2double(v); end
        M(:, j) = v;
    end
    Y_ahba_atlas(i, :) = nanmean(M, 1);
end

%% GTEx atlas: one row per unique parcel; row = mean of all GTEx samples at that parcel (all genes)
Y_gtex_atlas = nan(R_gtex, G_all);
xyz_gtex = zeros(R_gtex, 3);
for i = 1:R_gtex
    r = u_gtex(i);
    xyz_gtex(i, :) = X_target(r, :);
    rows = find(gtex & parcel_index == r);
    M = zeros(length(rows), G_all);
    for j = 1:G_all
        col = geneColIdx_all(j);
        v = T.(geneVarNames{col})(rows);
        if isnumeric(v), v = double(v); else, v = str2double(v); end
        M(:, j) = v;
    end
    Y_gtex_atlas(i, :) = nanmean(M, 1);
end

out = struct('T', T, 'Xcentroid', Xcentroid, 'parcel_index', parcel_index, ...
    'G', G, 'R', R, 'geneColIdx', geneColIdx, 'matchedNames', {matchedNames}, ...
    'X_target', X_target, 'targetParcelNames', {targetParcelNames}, 'geneVarNames', {geneVarNames}, ...
    'Y_subj_parcel_gene', Y_subj_parcel_gene, 'subjList', subjList, ...
    'Y_ahba_atlas', Y_ahba_atlas, 'xyz_ahba', xyz_ahba, 'ahba_site_parcel_idx', u_ahba, ...
    'Y_gtex_atlas', Y_gtex_atlas, 'xyz_gtex', xyz_gtex, 'gtex_site_parcel_idx', u_gtex, ...
    'G_atlas', G_all, 'geneColIdx_atlas', geneColIdx_all, 'matchedNames_atlas', {matchedNames_all});
end
