function [ hloss ] = cpLossToHost( dloss, opts )

for i = 1:numel(dloss.train.opts_out)
        hloss.train.opts_out{i} = gather(dloss.train.opts_out{i});
end

hloss.train.e       = gather(dloss.train.e);
hloss.train.e_errfun  = gather(dloss.train.e_errfun);
if opts.validation == 1
    hloss.val.e         = gather(dloss.val.e);
    hloss.val.e_errfun  = gather(dloss.val.e_errfun);
    for i = 1:numel(dloss.train.opts_out)
        hloss.val.opts_out{i} = gather(dloss.val.opts_out{i});
    end
end

end

