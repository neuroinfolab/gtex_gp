function [idx, matchedNames] = match_gene_names(tableVarNames, geneList)
% MATCH_GENE_NAMES Find table columns that match a list of gene names.
% Handles: case-insensitive, underscores vs hyphens/dots.
%
% idx = match_gene_names(T.Properties.VariableNames, hvgList)
% Returns column indices (into T) for genes in geneList that were found.
% matchedNames = variable names as in table for matched genes.

idx = [];
matchedNames = {};
tableNames = tableVarNames;
geneList = cellstr(geneList);

% Normalize for matching: upper case, replace hyphen with underscore
normTable = upper(strrep(tableNames, '-', '_'));
normTable = strrep(normTable, '.', '_');
for g = 1:length(geneList)
    gn = upper(strrep(geneList{g}, '-', '_'));
    gn = strrep(gn, '.', '_');
    found = find(strcmp(normTable, gn), 1);
    if isempty(found)
        % try without underscore (e.g. C1orf115)
        found = find(strcmpi(tableNames, geneList{g}), 1);
    end
    if isempty(found)
        % try contains
        for t = 1:length(tableNames)
            if contains(upper(tableNames{t}), upper(geneList{g})) && ...
                    length(tableNames{t}) <= length(geneList{g}) + 2
                found = t;
                break;
            end
        end
    end
    if ~isempty(found)
        idx(end+1) = found; %#ok<AGROW>
        matchedNames{end+1} = tableNames{found}; %#ok<AGROW>
    end
end
idx = idx(:);
matchedNames = matchedNames(:);
end
