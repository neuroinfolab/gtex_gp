function [Xcentroid, K, Xall] = parse_coordinates(coordCell)
% PARSE_COORDINATES Parse coordinate strings into 3D points.
% coordCell: Nx1 cell of strings like "[ (x,y,z), (x,y,z) ]".
% Returns Xcentroid (Nx3), K (# tuples per row), Xall (cell of Ki×3).

% Regex: capture (x,y,z) with optional whitespace and scientific numbers
pat = '\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)';
N = length(coordCell);
Xcentroid = nan(N, 3);
K = zeros(N, 1);
Xall = cell(N, 1);

for i = 1:N
    s = coordCell{i};
    if ischar(s)
        s = char(s);
    else
        s = char(string(s));
    end
    tok = regexp(s, pat, 'tokens');
    Ki = length(tok);
    K(i) = Ki;
    if Ki == 0
        continue;
    end
    % tok is 1×Ki cell; each tok{k} is 1×3 cell of strings → stack as Ki×3
    rows = cellfun(@(t) [str2double(t{1}), str2double(t{2}), str2double(t{3})], tok, 'UniformOutput', false);
    pts = vertcat(rows{:});
    valid = ~any(isnan(pts), 2);
    pts = pts(valid, :);
    if isempty(pts)
        continue;
    end
    Xall{i} = pts;
    Xcentroid(i, :) = mean(pts, 1);
end
end
