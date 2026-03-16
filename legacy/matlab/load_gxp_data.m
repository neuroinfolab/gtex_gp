function [T, dataDir] = load_gxp_data(csvPath)
% LOAD_GXP_DATA Load gxp_samples.csv using readtable (fast). Coordinates column is quoted.
% T = load_gxp_data()  % uses 'gxp_samples.csv' in same directory
% [T, dataDir] = load_gxp_data(csvPath)
%
% Returns table T with meta columns and gene columns (numeric). Coordinates kept as text.

if nargin < 1 || isempty(csvPath)
    csvPath = fullfile(fileparts(mfilename('fullpath')), 'gxp_samples.csv');
end
dataDir = fileparts(csvPath);
if isempty(dataDir)
    dataDir = pwd;
end

T = readtable(csvPath, 'VariableNamingRule', 'preserve', 'TextType', 'string');
% Ensure coordinates is cell of char for parse_coordinates (in case readtable returns string)
if isstring(T.coordinates)
    T.coordinates = cellstr(T.coordinates);
end
end
