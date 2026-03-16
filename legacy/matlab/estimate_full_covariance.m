% Estimate full (G*R)×(G*R) covariance from AHBA and GTEx using pairwise-complete observations.
% Saves out/covariance/full_covariance.mat and visualizations.
% Optional: [Sigma_full, mu] = estimate_full_covariance() or estimate_full_covariance(out) with out from load_and_format.
function [Sigma_full, mu] = estimate_full_covariance(out)
if nargin < 1, out = load_and_format(); end
root = fileparts(mfilename('fullpath'));
outDir = fullfile(root, 'out', 'covariance');
if ~exist(outDir, 'dir'), mkdir(outDir); end

T = out.T; parcel_index = out.parcel_index; G = out.G; R = out.R;
geneColIdx = out.geneColIdx; geneVarNames = out.geneVarNames;
nRow = height(T);
subj = string(T.subject);
[subjList, ~, subjIc] = unique(subj, 'stable');
nSubj = length(subjList);

%% Per-subject R×G matrices (mean over samples per parcel)
Y_cell = cell(nSubj, 1);
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
    Y_cell{i} = Y_i;
end

%% Stack to X: nSubj × (G*R), index k = (r-1)*G + g
GR = G * R;
X = nan(nSubj, GR);
for i = 1:nSubj
    Y = Y_cell{i};
    for r = 1:R
        for g = 1:G
            X(i, (r-1)*G + g) = Y(r, g);
        end
    end
end

%% Global mean and center
mu = nanmean(X, 1);
X_c = X - mu;
X_c(isnan(X)) = nan;

%% Pairwise-complete sample covariance
Sigma_full = zeros(GR, GR);
for i = 1:GR
    for j = i:GR
        mask = ~isnan(X_c(:,i)) & ~isnan(X_c(:,j));
        n_ij = sum(mask);
        if n_ij < 2, continue; end
        c = sum(X_c(mask,i) .* X_c(mask,j)) / (n_ij - 1);
        Sigma_full(i,j) = c;
        Sigma_full(j,i) = c;
    end
end

%% Ridge and symmetrize for numerical stability
lambda = 1e-6 * mean(diag(Sigma_full));
Sigma_full = Sigma_full + lambda * eye(GR);
Sigma_full = (Sigma_full + Sigma_full') / 2;

%% Save
targetParcelNames = out.targetParcelNames;
matchedNames = out.matchedNames;
save(fullfile(outDir, 'full_covariance.mat'), 'Sigma_full', 'mu', 'G', 'R', 'nSubj', ...
    'geneColIdx', 'targetParcelNames', 'matchedNames', '-v7.3');
fprintf('Saved %s (%d x %d)\n', fullfile(outDir, 'full_covariance.mat'), GR, GR);

%% Visualize
visualize_covariance(Sigma_full, G, R, outDir);
end

function visualize_covariance(Sigma_full, G, R, outDir)
GR = size(Sigma_full, 1);

% 1) R×R block summary (Frobenius norm of each G×G block)
block_norm = zeros(R, R);
for r = 1:R
    for rp = 1:R
        blk = Sigma_full((r-1)*G+1:r*G, (rp-1)*G+1:rp*G);
        block_norm(r, rp) = norm(blk, 'fro');
    end
end
figure('Visible', 'off');
imagesc(block_norm);
colorbar;
title('R x R block Frobenius norm (each block G x G)');
xlabel('Region'); ylabel('Region');
saveas(gcf, fullfile(outDir, 'full_cov_block_view.png'));
close(gcf);
fprintf('Saved full_cov_block_view.png\n');

% 2) Downsampled correlation heatmap (every 10th)
d = sqrt(diag(Sigma_full));
d(d < 1e-12) = 1;
Corr = Sigma_full ./ (d * d');
step = max(1, round(GR / 1000));
idx = 1:step:GR;
C_sub = Corr(idx, idx);
figure('Visible', 'off');
imagesc(C_sub);
colorbar;
title(sprintf('Correlation (downsampled 1:%d)', step));
saveas(gcf, fullfile(outDir, 'full_cov_correlation_downsampled.png'));
close(gcf);
fprintf('Saved full_cov_correlation_downsampled.png\n');

% 3) One gene slice: gene 1 across all R vs same (R×R)
g1 = 1;
idx_r = (0:R-1)*G + g1;
C_g1 = Sigma_full(idx_r, idx_r);
figure('Visible', 'off');
imagesc(C_g1);
colorbar;
title(sprintf('Covariance: gene %d across regions (R x R)', g1));
xlabel('Region'); ylabel('Region');
saveas(gcf, fullfile(outDir, 'full_cov_one_gene_slice.png'));
close(gcf);
fprintf('Saved full_cov_one_gene_slice.png\n');
end
