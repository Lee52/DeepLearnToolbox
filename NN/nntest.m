function [er, bad, opts_out] = nntest(nn, x, y)
    opts_out = [];
    labels = nnpredict(nn, x);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
