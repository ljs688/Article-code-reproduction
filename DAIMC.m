function [U,V,B,F,P,N] = DAIMC(X,W,U,V,B,label,r,viewNum,options)
% If you use the code, please cite the following papers:
% [1] Hu M, Chen S. Doubly aligned incomplete multi-view clustering[C]//IJCAI 2018.
% [2] Jie Wen et al., IEEE TSMC, 2022.

eta = 1e-10;
if isfield(options, 'tol')
    tol = options.tol;
else
    tol = 1e-4;
end
if isfield(options, 'outerMaxIter')
    outerMaxIter = options.outerMaxIter;
else
    outerMaxIter = 30;
end
if isfield(options, 'maxRunSeconds')
    maxRunSeconds = options.maxRunSeconds;
else
    maxRunSeconds = inf;
end
if isfield(options, 'verboseDAIMC')
    verboseDAIMC = options.verboseDAIMC;
else
    verboseDAIMC = false;
end
F = 0;
P = 0;
N = 0;
D = cell(viewNum,1);
for i = 1:viewNum
    for k = 1:size(B{i},1)
        D{i}(k,k) = 1/sqrt(norm(B{i}(k,:),2).^2+eta);
    end
end

runTimer = tic;
time = 0;
f = 0;
while 1
    time = time+1;
    if verboseDAIMC
        fprintf('            DAIMC iter %d (elapsed %.2fs)\n', time, toc(runTimer));
        drawnow;
    end
    for i = 1:viewNum
        tmp1 = options.afa*B{i}*B{i}';
        tmp2 = V'*W{i}*V;
        tmp3 = X{i}*W{i}*V + options.afa*B{i};
        U{i} = lyap(tmp1, tmp2, -tmp3);
    end
    V = UpdateV_DAIMC(X,W,U,V,viewNum,options);
    Q = diag(ones(1,size(V,1))*V);
    V = V * inv(Q);
    for i = 1:viewNum
        U{i} = U{i}*Q;
        invD = diag(1./diag(0.5*options.beta*D{i}));
        B{i} = (invD - invD * U{i} * inv(U{i}'*invD*U{i} + eye(r)) * U{i}' * invD)*U{i};
        for k = 1:size(B{i},1)
            D{i}(k,k) = 1/sqrt(norm(B{i}(k,:),2).^2+eta);
        end
    end

    ff = 0;
    for i = 1:viewNum
        tmp1 = (X{i} - U{i}*V')*W{i};
        tmp2 = B{i}'*U{i} - eye(r);
        tmp3 = sum(1./diag(D{i}));
        ff = ff + sum(sum(tmp1.^2)) + options.afa*(sum(sum(tmp2.^2)) + options.beta*tmp3);
    end
    F(time) = ff;
    if isnan(ff) || isinf(ff) || abs(ff-f)/max(abs(f),1e-10) < tol || abs(ff-f) > 1e100 || time >= outerMaxIter || toc(runTimer) >= maxRunSeconds
        break;
    end
    f = ff;
end
end
