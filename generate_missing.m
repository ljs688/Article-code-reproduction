function generate_missing(Dataname, percentDel, numFolds, seed)
% generate_missing: 为数据集生成缺失索引文件

if nargin < 3, numFolds = 5;  end
if nargin < 4, seed     = 42; end

rng(seed);
tmp = load(Dataname);

% 读取特征（兼容不同变量名）
if isfield(tmp, 'X')
    X = tmp.X;
elseif isfield(tmp, 'data')
    X = tmp.data;
else
    error('数据集 %s 中找不到特征变量 (X 或 data)', Dataname);
end

% 读取标签（兼容不同变量名）
if isfield(tmp, 'gt')
    truth = tmp.gt;
elseif isfield(tmp, 'truelabel')
    truth = tmp.truelabel;
elseif isfield(tmp, 'Y')
    truth = tmp.Y;
elseif isfield(tmp, 'label')
    truth = tmp.label;
elseif isfield(tmp, 'truth')
    truth = tmp.truth;
else
    error('数据集 %s 中找不到标签变量', Dataname);
end

% 若标签是 cell，先转为普通数组
if iscell(truth), truth = cell2mat(truth); end
truth    = double(truth(:));
numInst  = length(truth);
num_view = length(X);

% 确保样本在列方向
if size(X{1}, 2) ~= numInst
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end

folds = cell(numFolds, 1);

for f = 1:numFolds
    ind = ones(numInst, num_view);

    for iv = 1:num_view
        numDel = round(numInst * percentDel);
        delIdx = randperm(numInst, numDel);
        ind(delIdx, iv) = 0;
    end

    % 修复全缺失样本：至少保留一个视图的观测
    allMissing = find(sum(ind, 2) == 0);
    for k = 1:length(allMissing)
        v = randi(num_view);
        ind(allMissing(k), v) = 1;
    end

    folds{f} = ind;
end

saveName = [Dataname, '_percentDel_', num2str(percentDel), '.mat'];
save(saveName, 'folds');
fprintf('已保存: %s\n', saveName);
fprintf('  样本数: %d, 视图数: %d, 缺失率: %.0f%%, 折数: %d\n', ...
    numInst, num_view, percentDel*100, numFolds);
end
