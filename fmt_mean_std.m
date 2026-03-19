function s = fmt_mean_std(v)
    if isempty(v)
        s = '  N/A';
    else
        s = sprintf('%5.2f+-%5.2f', mean(v), std(v));
    end
end
