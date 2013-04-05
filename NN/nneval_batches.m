function loss = nneval_batches(nn, loss,train_x, train_y, val_x, val_y)
%wrapper for nneval to add support for batch evalulation of perfomance
assert(nargin == 4 || nargin == 6, 'Wrong number of arguments');
if nn.isGPU  % check if neural network is on gpu or not
    nnfeedforward = @nnff_gpu;
else
    nnfeedforward = @nnff;
end

nbatch_train = length(train_x);
nbatch_val = length(val_x);

mtrain = 0;
for i=1:nbatch_train
    mtrain = mtrain+length(train_x{i});
end

mval = 0;
for i=1:nbatch_val
    mval = mval+length(val_x{i});
end


Ltrain = 0;
Lval = 0;

for i = 1:nbatch_train
    if nn.isGPU
        tx = gpuArray(train_x{i});
        ty = gpuArray(train_x{i});
    else
        tx = train_x{i};
        ty = train_x{i}; 
    end
    nn_new     = nnfeedforward(nn, tx, ty);
    current_batch_size = length(tx);
    Ltrain = Ltrain + nn_new.L*(current_batch_size/mtrain);
    clear tx ty
end
loss.train.e = [loss.train.e; Ltrain];

if nargin == 6
    for i = 1:nbatch_val
        if nn.isGPU
            vx = gpuArray(val_x{i});
            vy = gpuArray(val_x{i});
        else
            vx = val_x{i};
            vy = val_x{i};
        end
        
        nn_new     = nnfeedforward(nn, vx, vy);
        current_batch_size = length(vx);
        Lval = Lval + nn_new.L*(current_batch_size/mval);
        clear vx vy
    end
end
loss.val.e   = [loss.val.e; Lval];

%If error function is supplied apply it
if ~isempty(nn.errfun)
    
    for i = 1:nbatch_train
       [er_train, ~]               = nn.errfun(nn, train_x{i}, train_y{i});
       loss.train.e_errfun         = [loss.train.e_errfun; er_train];
    end
    
    if nargin == 6
        for i=1:nbatch_val
            [er_val, ~]             = nn.errfun(nn, val_x{i}, val_y{i});
            loss.val.e_errfun      = [loss.val.e_errfun; er_val];
        end
    end
end

end