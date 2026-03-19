% =========================================================
% run_DAIMC_all.m
% =========================================================
clear; clc;

%% -------- config --------
dataNames = {
    'MSRCV1', ...
    '2view-caltech101-8677sample', ...
    'Hdigit', ...
    'YouTubeFace'
};

seeds = [0, 1, 2, 3, 4];
percentDel = 0.5;
fixedFoldIdx = 1;   % keep missing pattern fixed across seeds

options.afa = 0.0001;
options.beta = 10000;
options.outerMaxIter = 12;
options.updateVMaxIter = 10;
options.tol = 1e-4;
options.maxRunSeconds = 45;
options.newinitReplicates = 5;
options.newinitMaxIter = 40;
options.finalReplicates = 3;
options.verboseDAIMC = true;
%% ------------------------

numData = length(dataNames);
numSeeds = length(seeds);
totalRuns = numData * numSeeds;
doneRuns = 0;

allResults = nan(numData, numSeeds, 3);

rawFile = 'DAIMC_raw.txt';
fraw = fopen(rawFile, 'a');
fprintf(fraw, '\n===== Run start: %s =====\n', datestr(now));
fprintf(fraw, '%-40s  %-6s  %-8s  %-8s  %-8s\n', 'Dataset', 'Seed', 'ACC', 'NMI', 'ARI');
fclose(fraw);

totalTimer = tic;

for di = 1:numData
    Dataname = dataNames{di};
    Datafold = [Dataname, '_percentDel_', num2str(percentDel), '.mat'];

    fprintf('\n[%d/%d Dataset] %s\n', di, numData, Dataname);
    fprintf('--------------------------------------------\n');

    tmp = load(Dataname);

    if isfield(tmp, 'X')
        X_raw = tmp.X;
    elseif isfield(tmp, 'data')
        X_raw = tmp.data;
    else
        error('Cannot find feature variable (X or data) in %s.mat', Dataname);
    end

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
        error('Cannot find label variable in %s.mat', Dataname);
    end
    if iscell(truth)
        truth = cell2mat(truth);
    end
    truth = double(truth(:));

    tmp2 = load(Datafold);
    folds = tmp2.folds;
    numFolds = numel(folds);
    if fixedFoldIdx < 1 || fixedFoldIdx > numFolds
        error('fixedFoldIdx=%d out of range. Available folds: 1..%d', fixedFoldIdx, numFolds);
    end
    ind_folds_fixed = double(folds{fixedFoldIdx});

    if ~iscell(X_raw)
        if isstruct(X_raw)
            fields = fieldnames(X_raw);
            tmp_cell = cell(length(fields), 1);
            for kk = 1:length(fields)
                tmp_cell{kk} = X_raw.(fields{kk});
            end
            X_raw = tmp_cell;
        else
            X_raw = {X_raw};
        end
    end

    num_view = length(X_raw);
    numInst = length(truth);

    if size(ind_folds_fixed, 2) ~= num_view
        error('fold view count mismatch: folds has %d views but data has %d views', ...
            size(ind_folds_fixed, 2), num_view);
    end

    % Align each view independently so columns are samples.
    viewInst = zeros(num_view, 1);
    for iv = 1:num_view
        Xi = double(X_raw{iv});
        [ri, ci] = size(Xi);
        if ci == numInst
            % already d x n
        elseif ri == numInst
            Xi = Xi';
            [ri, ci] = size(Xi);
        else
            % fallback: choose orientation whose sample count is closer to truth
            if abs(ci - numInst) > abs(ri - numInst)
                Xi = Xi';
                [ri, ci] = size(Xi);
            end
        end
        X_raw{iv} = Xi;
        viewInst(iv) = size(Xi, 2);
    end

    commonInst = min([numInst; viewInst; size(ind_folds_fixed, 1)]);
    if commonInst < 2
        error('invalid common sample count after alignment: %d', commonInst);
    end
    if commonInst ~= numInst || any(viewInst ~= commonInst) || size(ind_folds_fixed, 1) ~= commonInst
        fprintf('  [align] truth=%d, views=%s, foldRows=%d -> common=%d\n', ...
            numInst, mat2str(viewInst'), size(ind_folds_fixed, 1), commonInst);
        truth = truth(1:commonInst);
        ind_folds_fixed = ind_folds_fixed(1:commonInst, :);
        for iv = 1:num_view
            X_raw{iv} = X_raw{iv}(:, 1:commonInst);
        end
        numInst = commonInst;
    end

    numClust = length(unique(truth));

    fprintf('  samples: %d  views: %d  classes: %d\n', numInst, num_view, numClust);
    fprintf('  fixed fold idx: %d / %d (same fold for all seeds)\n', fixedFoldIdx, numFolds);
    obsRates = mean(ind_folds_fixed, 1);
    for iv = 1:num_view
        fprintf('    view %d observed ratio: %.2f%% (inst=%d)\n', iv, obsRates(iv) * 100, size(X_raw{iv}, 2));
    end

    dataTimer = tic;

    for si = 1:numSeeds
        seed = seeds(si);
        doneRuns = doneRuns + 1;

        elapsed = toc(totalTimer);
        if doneRuns > 1
            eta = elapsed / (doneRuns - 1) * (totalRuns - doneRuns + 1);
            fprintf('  [progress %2d/%d] seed %d | elapsed %s | eta %s\n', ...
                doneRuns, totalRuns, seed, fmt_time(elapsed), fmt_time(eta));
        else
            fprintf('  [progress %2d/%d] seed %d | elapsed %s\n', ...
                doneRuns, totalRuns, seed, fmt_time(elapsed));
        end

        runTimer = tic;
        rng(seed);
        ind_folds = ind_folds_fixed;

        X = cell(num_view, 1);
        W = cell(num_view, 1);
        for iv = 1:num_view
            X1 = double(X_raw{iv});
            X1 = NormalizeFea(X1, 0);
            ind_0 = find(ind_folds(:, iv) == 0);
            X1(:, ind_0) = 0;
            X{iv} = X1;
            W{iv} = diag(double(ind_folds(:, iv)));
        end

        try
            fprintf('         stage: newinit ...\n'); drawnow;
            tInit = tic;
            [U0, V0, B0] = newinit(X, W, numClust, num_view, options);
            initCost = toc(tInit);
            fprintf('         stage: newinit done (%.2fs)\n', initCost); drawnow;

            fprintf('         stage: DAIMC ...\n'); drawnow;
            tCore = tic;
            [U, V, B, ~, ~, ~] = DAIMC(X, W, U0, V0, B0, truth, numClust, num_view, options);
            coreCost = toc(tCore);
            fprintf('         stage: DAIMC done (%.2fs)\n', coreCost); drawnow;

            fprintf('         stage: kmeans ...\n'); drawnow;
            tKm = tic;
            indic = kmeans(V, numClust, 'Replicates', options.finalReplicates, 'MaxIter', 100);
            kmCost = toc(tKm);
            fprintf('         stage: kmeans done (%.2fs)\n', kmCost); drawnow;

            fprintf('         stage: eval ClusteringMeasure ...\n'); drawnow;
            res = ClusteringMeasure(truth, indic) * 100;
            acc = res(1);
            nmi = res(2);
            fprintf('         stage: eval RandIndex ...\n'); drawnow;
            [ari, ~, ~, ~] = RandIndex(truth, indic);
            ari = ari * 100;
            fprintf('         stage: eval done\n'); drawnow;

            allResults(di, si, 1) = acc;
            allResults(di, si, 2) = nmi;
            allResults(di, si, 3) = ari;

            fprintf('         result: ACC=%.2f  NMI=%.2f  ARI=%.2f  (time %s)\n', ...
                acc, nmi, ari, fmt_time(toc(runTimer)));
            fprintf('               stage time: init=%.2fs, daimc=%.2fs, kmeans=%.2fs\n', ...
                initCost, coreCost, kmCost);

            fprintf('         stage: write raw log ...\n'); drawnow;
            fraw = fopen(rawFile, 'a');
            fprintf(fraw, '%-40s  %-6d  %-8.2f  %-8.2f  %-8.2f\n', Dataname, seed, acc, nmi, ari);
            fclose(fraw);
            fprintf('         stage: write raw log done\n'); drawnow;

        catch err
            fprintf('         failed: %s\n', err.message);
            fraw = fopen(rawFile, 'a');
            fprintf(fraw, '%-40s  %-6d  ERROR: %s\n', Dataname, seed, err.message);
            fclose(fraw);
        end
    end

    fprintf('  >> %s all seeds done, time %s\n', Dataname, fmt_time(toc(dataTimer)));
end

%% -------- summary output --------
fprintf('\n\n============================================================\n');
fprintf('%-40s  %-14s  %-14s  %-14s\n', 'Dataset', 'ACC', 'NMI', 'ARI');
fprintf('------------------------------------------------------------\n');

summaryTable = cell(numData, 4);

for di = 1:numData
    row_acc = allResults(di, :, 1); row_acc = row_acc(~isnan(row_acc));
    row_nmi = allResults(di, :, 2); row_nmi = row_nmi(~isnan(row_nmi));
    row_ari = allResults(di, :, 3); row_ari = row_ari(~isnan(row_ari));

    str_acc = fmt_mean_std(row_acc);
    str_nmi = fmt_mean_std(row_nmi);
    str_ari = fmt_mean_std(row_ari);

    fprintf('%-40s  %-14s  %-14s  %-14s\n', dataNames{di}, str_acc, str_nmi, str_ari);

    summaryTable{di,1} = dataNames{di};
    summaryTable{di,2} = str_acc;
    summaryTable{di,3} = str_nmi;
    summaryTable{di,4} = str_ari;
end
fprintf('============================================================\n');
fprintf('total time: %s\n', fmt_time(toc(totalTimer)));

save('DAIMC_results.mat', 'allResults', 'summaryTable', 'dataNames', 'seeds');

fsum = fopen('DAIMC_summary.txt', 'w');
fprintf(fsum, 'DAIMC summary\n');
fprintf(fsum, 'missing rate: %.0f%%, seeds: 0~4\n', percentDel*100);
fprintf(fsum, 'total time: %s\n', fmt_time(toc(totalTimer)));
fprintf(fsum, '============================================================\n');
fprintf(fsum, '%-40s  %-14s  %-14s  %-14s\n', 'Dataset', 'ACC', 'NMI', 'ARI');
fprintf(fsum, '------------------------------------------------------------\n');
for di = 1:numData
    fprintf(fsum, '%-40s  %-14s  %-14s  %-14s\n', ...
        summaryTable{di,1}, summaryTable{di,2}, summaryTable{di,3}, summaryTable{di,4});
end
fprintf(fsum, '============================================================\n');
fclose(fsum);

fprintf('\nSaved summary to DAIMC_summary.txt\n');
fprintf('Saved per-run raw logs to DAIMC_raw.txt\n');
