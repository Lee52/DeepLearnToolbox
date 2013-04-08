function [ dloss ] = cpLossToGPU( hloss, opts )

dloss.train.e           = gpuArray(hloss.train.e);
dloss.train.e_errfun    = gpuArray(hloss.train.e_errfun);

for i = 1:numel(hloss.train.opts_out)
        dloss.train.opts_out{i} = gpuArray(hloss.train.opts_out{i});
end
if opts.validation == 1
    dloss.val.e         = gpuArray(hloss.val.e);
    dloss.val.e_errfun  = gpuArray(hloss.val.e_errfun);
    for i = 1:numel(hloss.train.opts_out)
        dloss.val.opts_out{i} = gpuArray(hloss.val.opts_out{i});
    end
end

end

