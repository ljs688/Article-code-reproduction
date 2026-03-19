function [AR, RI, MI, HI] = RandIndex(c1, c2)
% RandIndex: 计算 Adjusted Rand Index (ARI) 及相关指标
% 输入:
%   c1 - 真实标签向量
%   c2 - 聚类结果向量
% 输出:
%   AR - Adjusted Rand Index (ARI)
%   RI - Rand Index
%   MI - Mirkin Index
%   HI - Hubert Index

c1 = c1(:);
c2 = c2(:);
n  = length(c1);

u1 = unique(c1);  u2 = unique(c2);
r  = length(u1);  s  = length(u2);

% 列联表
T = zeros(r, s);
for i = 1:r
    for j = 1:s
        T(i,j) = sum(c1==u1(i) & c2==u2(j));
    end
end

a = sum(T, 2);
b = sum(T, 1);

nc2 = @(x) x.*(x-1)/2;

sumT  = sum(sum(nc2(T)));
sumA  = sum(nc2(a));
sumB  = sum(nc2(b));
total = nc2(n);

expected = sumA * sumB / total;
maxIdx   = (sumA + sumB) / 2;

if maxIdx == expected
    AR = 1;
else
    AR = (sumT - expected) / (maxIdx - expected);
end

TP = sumT;
FP = sumA - TP;
FN = sumB - TP;
TN = total - TP - FP - FN;

RI = (TP + TN) / total;
MI = (FP + FN) / total;
HI = (TP - FP - FN + TN) / total;
end
