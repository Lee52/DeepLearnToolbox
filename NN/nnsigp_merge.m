function [err] = nnsigp_merge(opts_out,batch_size)   
    cfmat = zeros(size(opts_out{1}));
    for j = 1:numel(opts_out)
        cfmat = cfmat + opts_out{j};
    end

    err(1) = matthew(cfmat(:,:,1));       % 1) signalpeptide(1) MCC
    err(2) = specificity(cfmat(:,:,2));   % 2) Cleavage site(2) specificity
    err(3) = precision(cfmat(:,:,2));   % 3) Cleavage site(2) precision
    err(4) = matthew(cfmat(:,:,2));   % 4) Cleavage site(2) MCC
    err(5) = matthew(cfmat(:,:,3));     % 5) transmembrane(3) MCC
end