function [assignment, cost] = Hungarian(costMat)
% Hungarian algorithm (Munkres) for optimal assignment
% 输入: costMat - n x m 代价矩阵（最小化总代价）
% 输出:
%   assignment - 1 x n 向量，assignment(i) = j 表示第i行分配给第j列
%   cost       - 最优总代价

costMat(isnan(costMat)) = Inf;
validMat = costMat(~isinf(costMat));
if isempty(validMat)
    error('costMat contains only Inf');
end

[n, m] = size(costMat);
dMat = costMat;

% 扩展为方阵
if n > m
    dMat(:, end+1:n) = max(dMat(~isinf(dMat)))*10;
    m = n;
elseif n < m
    dMat(end+1:m, :) = max(dMat(~isinf(dMat)))*10;
    n = m;
end

% Step 1: 每行减去行最小值
dMat = bsxfun(@minus, dMat, min(dMat,[],2));
% Step 2: 每列减去列最小值
dMat = bsxfun(@minus, dMat, min(dMat,[],1));

starZ  = zeros(n);
primeZ = zeros(n);
covRow = false(n,1);
covCol = false(1,n);

% Step 3: 找零元素并标星（每行每列最多一个）
for i = 1:n
    for j = 1:n
        if dMat(i,j)==0 && ~any(starZ(i,:)) && ~any(starZ(:,j)')
            starZ(i,j) = 1;
        end
    end
end

while true
    % Step 4: 覆盖含星零的列
    covCol = any(starZ, 1);
    if sum(covCol) >= n
        break;
    end
    
    % Step 5: 找未覆盖零并标撇
    done = false;
    while ~done
        [rPrime, cPrime] = find(dMat==0 & ~covRow & ~covCol, 1);
        if isempty(rPrime)
            done = true;
            break;
        end
        primeZ(rPrime, cPrime) = 1;
        starCol = find(starZ(rPrime,:));
        if isempty(starCol)
            % Step 6: 增广路径
            path = [rPrime, cPrime];
            while true
                r2 = find(starZ(:, path(end,2)));
                if isempty(r2), break; end
                path(end+1,:) = [r2, path(end,2)];
                c2 = find(primeZ(r2,:));
                path(end+1,:) = [r2, c2];
            end
            for k = 1:size(path,1)
                if mod(k,2)==1
                    starZ(path(k,1), path(k,2)) = 1;
                else
                    starZ(path(k,1), path(k,2)) = 0;
                end
            end
            primeZ(:) = 0;
            covRow(:) = false;
            covCol(:) = false;
            done = true;
        else
            covRow(rPrime) = true;
            covCol(starCol) = false;
        end
    end
    
    if ~done, continue; end
    
    % Step 7: 找未覆盖最小值并调整矩阵
    uncov = dMat(~covRow, ~covCol);
    minVal = min(uncov(:));
    if isinf(minVal), error('No feasible assignment'); end
    dMat(covRow,  :)     = dMat(covRow,  :)     + minVal;
    dMat(:,  ~covCol)    = dMat(:,  ~covCol)    - minVal;
end

% 提取结果
[r, c] = find(starZ);
assignment = zeros(1, size(costMat,1));
for k = 1:length(r)
    if r(k) <= size(costMat,1) && c(k) <= size(costMat,2)
        assignment(r(k)) = c(k);
    end
end

cost = sum(costMat(sub2ind(size(costMat), ...
    find(assignment>0), assignment(assignment>0))));
end
