function loss = nnpreallocateloss(nn,opts,train_x,train_y)
if ~isempty(nn.errfun)   %determine number of returned error values
    nerrfun =  numel(nn.errfun(nn, train_x(1,:), train_y(1,:)));
    loss.train.e_errfun        = zeros(opts.numepochs,nerrfun);
    if opts.validation
        loss.val.e_errfun        = zeros(opts.numepochs,nerrfun);
    else
        loss.val.e_errfun        = [];
    end
    
    [~, ~, opts_out] = nn.errfun(nn, train_x(1,:), train_y(1,:));
    for i = 1:opts.numepochs
        loss.train.opts_out{i} = zeros(size(opts_out));
        if opts.validation
            loss.val.opts_out{i}   = zeros(size(opts_out));
        else
            loss.val.opts_out = {};
        end
    end
    
else
    loss.val.e_errfun          = [];
    loss.train.e_errfun        = [];
    loss.val.opts_out          = {};
    loss.train.opts_out        = {};
end

loss.train.e               = zeros(opts.numepochs,1);
loss.val.e                 = zeros(opts.numepochs,1);
end