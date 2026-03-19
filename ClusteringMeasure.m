function result = ClusteringMeasure(label, indic)
% ClusteringMeasure: 璁＄畻鑱氱被璇勪及鎸囨爣
% 杈撳叆:
%   label  - 鐪熷疄鏍囩鍚戦噺 (n x 1)
%   indic  - 鑱氱被缁撴灉鍚戦噺 (n x 1)
% 杈撳嚭:
%   result - [ACC, NMI, Purity]  (0~1 鑼冨洿锛岃皟鐢ㄥ *100 杞櫨鍒嗘瘮)

label = label(:);
indic = indic(:);

ACC    = compute_acc(label, indic);
NMI    = compute_nmi(label, indic);
Purity = compute_purity(label, indic);

result = [ACC, NMI, Purity];
end

% ---------------------------------------------------------
function acc = compute_acc(label, indic)
label = label(:);  indic = indic(:);
uL = unique(label);  uC = unique(indic);
nL = length(uL);     nC = length(uC);
k  = max(nL, nC);

% 娣锋穯鐭╅樀
D = zeros(k, k);
for i = 1:length(label)
    ri = find(uL == label(i));
    ci = find(uC == indic(i));
    D(ri, ci) = D(ri, ci) + 1;
end

assignment = solve_assignment_max(D);
acc = 0;
for i = 1:nL
    if assignment(i) ~= 0
        acc = acc + D(i, assignment(i));
    end
end
acc = acc / length(label);
end

% ---------------------------------------------------------
function assignment = solve_assignment_max(D)
[nR, nC] = size(D);
k = max(nR, nC);
if nR ~= nC
    D(k, k) = 0;  % zero padding
end

if k <= 8
    p = perms(1:k);
    scores = zeros(size(p,1), 1);
    for t = 1:size(p,1)
        idx = sub2ind([k, k], (1:k)', p(t,:)');
        scores(t) = sum(D(idx));
    end
    [~, best] = max(scores);
    assignment = p(best, :);
else
    [assignment, ~] = Hungarian(-D);
end

if numel(assignment) < nR
    assignment(numel(assignment)+1:nR) = 0;
end
assignment = assignment(1:nR);
end

% ---------------------------------------------------------
function nmi = compute_nmi(label, indic)
label = label(:);  indic = indic(:);
n  = length(label);
uL = unique(label);  uC = unique(indic);

MI = 0;
for i = 1:length(uL)
    for j = 1:length(uC)
        nij = sum(label == uL(i) & indic == uC(j));
        ni  = sum(label == uL(i));
        nj  = sum(indic == uC(j));
        if nij > 0
            MI = MI + (nij/n) * log2((nij*n) / (ni*nj));
        end
    end
end

H_L = 0;
for i = 1:length(uL)
    p = sum(label == uL(i)) / n;
    if p > 0, H_L = H_L - p*log2(p); end
end
H_C = 0;
for j = 1:length(uC)
    p = sum(indic == uC(j)) / n;
    if p > 0, H_C = H_C - p*log2(p); end
end

denom = (H_L + H_C);
if denom == 0, nmi = 0; else, nmi = max(0, 2*MI/denom); end
end

% ---------------------------------------------------------
function purity = compute_purity(label, indic)
label = label(:);  indic = indic(:);
n  = length(label);
uC = unique(indic);
correct = 0;
for j = 1:length(uC)
    sub = label(indic == uC(j));
    correct = correct + max(histc(sub, unique(sub)));
end
purity = correct / n;
end

% ---------------------------------------------------------
% 鍖堢墮鍒╃畻娉曪紙Munkres锛夛紝姹傛柟褰唬浠风煩闃电殑鏈€灏忔潈瀹岀編鍖归厤
% ---------------------------------------------------------
function assignment = munkres(C)
[n, m] = size(C);
padded = false;
if n ~= m
    sz = max(n, m);
    C(sz, sz) = 0;   % 闆跺～鍏?    padded = true;
end
sz = size(C, 1);

% 琛屽垪瑙勭害
C = bsxfun(@minus, C, min(C,[],2));
C = bsxfun(@minus, C, min(C,[],1));

star  = false(sz);
prime = false(sz);
rCov  = false(sz, 1);
cCov  = false(1, sz);

% 鍒濆鏄熷寲
for i = 1:sz
    for j = 1:sz
        if C(i,j)==0 && ~any(star(i,:)) && ~any(star(:,j)')
            star(i,j) = true;
        end
    end
end

while true
    cCov = any(star, 1);
    if sum(cCov) >= sz, break; end

    changed = true;
    while changed
        changed = false;
        [ri, ci] = find(C==0 & ~prime & ...
            bsxfun(@or, rCov, cCov)==0);   % 鏈鐩栫殑0
        if isempty(ri), break; end

        for k = 1:length(ri)
            i = ri(k);  j = ci(k);
            if rCov(i) || cCov(j), continue; end
            prime(i,j) = true;
            sc = find(star(i,:), 1);
            if isempty(sc)
                % 澧炲箍
                r = i;  c = j;
                while true
                    sr = find(star(:,c), 1);
                    if isempty(sr), break; end
                    star(sr, c) = false;
                    pc = find(prime(sr,:), 1);
                    star(sr, pc) = true;
                    c = pc;
                end
                star(r, j) = true;
                prime(:) = false;
                rCov(:)  = false;
                cCov = any(star, 1);
                changed = true;
                break;
            else
                rCov(i)  = true;
                cCov(sc) = false;
                changed  = true;
            end
        end
    end

    % 璋冩暣
    uncov = C(~rCov, ~cCov);
    h = min(uncov(:));
    if isempty(h) || isinf(h), break; end
    C(rCov, :)   = C(rCov, :)   + h;
    C(:, ~cCov)  = C(:, ~cCov)  - h;  % 绛変环浜庡噺鍘绘湭瑕嗙洊鍒楋紝鍐嶅姞鍥炶鐩栬
    % 姝ｇ‘鐗堟湰:
    C = C + h * (rCov * ones(1,sz));
    C = C - h * (ones(sz,1) * ~cCov);
end

assignment = zeros(1, n);
for i = 1:n
    j = find(star(i,:), 1);
    if ~isempty(j) && j <= m
        assignment(i) = j;
    end
end
end

