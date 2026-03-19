function X = NormalizeFea(X, row)
% NormalizeFea: 特征归一化
%   NormalizeFea(X, 0) 对每一列（样本）做 L2 归一化
%   NormalizeFea(X, 1) 对每一行（特征维度）做 L2 归一化
%
% 输入:
%   X   - d x n 矩阵（行=特征维, 列=样本）
%   row - 0: 按列归一化（默认）; 1: 按行归一化

if nargin < 2
    row = 0;
end

if row == 0
    % 按列（样本）归一化
    norms = sqrt(sum(X.^2, 1));   % 1 x n
    norms(norms == 0) = 1;        % 防止除零
    X = bsxfun(@rdivide, X, norms);
else
    % 按行（特征维）归一化
    norms = sqrt(sum(X.^2, 2));   % d x 1
    norms(norms == 0) = 1;
    X = bsxfun(@rdivide, X, norms);
end
end
