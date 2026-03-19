function s = fmt_time(sec)
    if sec < 60
        s = sprintf('%.0fs', sec);
    elseif sec < 3600
        s = sprintf('%.0fm%.0fs', floor(sec/60), mod(sec,60));
    else
        s = sprintf('%.0fh%.0fm', floor(sec/3600), floor(mod(sec,3600)/60));
    end
end
