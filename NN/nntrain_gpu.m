function [hnn, L,hloss]  = nntrain_gpu(hnn, htrain_x, htrain_y, opts, hval_x, hval_y)
%NNTRAIN trains a neural net on cpu
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.
%
% hVARNAME is a variable on the host
% dVARNAME is a varibale on the gpu device
gpu = gpuDevice();
reset(gpu);
wait(gpu);
disp(['GPU memory available (Gb): ', num2str(gpu.FreeMemory / 10^9)]);
cast = hnn.cast;
caststr = hnn.caststr;
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')
hnn.isGPU = 0; % tell code that variables are not on gpu (this is the HOSTnn)
dnn = cpNNtoGPU(hnn,cast);   % COPY NETWORK TO DEVICE, cpNNtoGPU sets dnn.isGPU = 1
m = size(htrain_x, 1);
spec_prec_mean_old = 0;

%divide training set into batches to fit GPU memory
[htrainbatches_x, htrainbatches_y] = nnevaldata2batches(opts,htrain_x,htrain_y);


if nargin == 6
    opts.validation = 1;
    [hvalbatches_x, hvalbatches_y] = nnevaldata2batches(opts,hval_x,hval_y);
else
    opts.validation = 0;
end

%initialize loss structs
hloss = nnpreallocateloss(hnn,opts,htrain_x,htrain_y);
dloss = cpLossToGPU(hloss, opts);  %copy preallocated loss struct to gpu


fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
    %check if plotting function is supplied, else use nnupdatefigures
    if ~isfield(opts,'plotfun')  || isempty(opts.plot)
        opts.plotfun = @nnupdatefigures;
    end
    
end

if isfield(opts, 'outputfolder') && ~isempty(opts.outputfolder)
    save_nn_flag = 1;
else
    save_nn_flag = 0;
end

%variable momentum
if isfield(opts, 'momentum_variable') && ~isempty(opts.momentum_variable)
    if length(opts.momentum_variable) ~= opts.numepochs
        error('opts.momentum_variable must specify a momentum value for each epoch ie length(opts.momentum_variable) == opts.numepochs')
    end
    var_momentum_flag = 1;
else
    var_momentum_flag = 0;
end

%variable learningrate
if isfield(opts, 'learningRate_variable') && ~isempty(opts.learningRate_variable)
    if length(opts.learningRate_variable) ~= opts.numepochs
        error('opts.learningRate_variable must specify a learninrate value for each epoch ie length(opts.learningRate_variable) == opts.numepochs')
    end
    var_learningRate_flag = 1;
else
    var_learningRate_flag = 0;
end

batchsize = opts.batchsize;
numepochs = opts.numepochs;
numbatches = floor(m / batchsize);
L = zeros(numepochs*numbatches,1);
n = 1;
              



for i = 1 : numepochs
    epochtime = (tic);
    %update momentum
    if var_momentum_flag
        hnn.momentum = opts.momentum_variable(i);
        dnn.momentum = opts.momentum_variable(i);
    end
    %update learning rate
    if var_learningRate_flag
        hnn.learningRate = opts.learningRate_variable(i);
        dnn.learningRate = opts.learningRate_variable(i);
    end
    
    kk = randperm(m);
    for l = 1 : numbatches
        
        hbatch_x = extractminibatch(kk,l,batchsize,htrain_x);
        
        %Add noise to input (for use in denoising autoencoder)
        if(hnn.inputZeroMaskedFraction ~= 0)
            hbatch_x = hbatch_x.*(gpuArray.rand(size(hbatch_x),caststr)>hnn.inputZeroMaskedFraction);
        end
        
        % COPY BATCHES TO GPU DEVICE
        dbatch_x = gpuArray(cast(hbatch_x));
        dbatch_y = gpuArray(cast(extractminibatch(kk,l,batchsize,htrain_y)));
        
        % use gpu functions to train
        dnn = nnff_gpu(dnn, dbatch_x, dbatch_y);
        dnn = nnbp_gpu(dnn);
        dnn = nnapplygrads(dnn);
        L(n) = gather(dnn.L);
        n = n + 1;
    end
    
    t = toc(epochtime);
    evalt = tic;
    

    %after each epoch update losses
    if opts.validation == 1
        dloss = nneval_batches(dnn, opts, dloss, i, htrainbatches_x, htrainbatches_y, hvalbatches_x, hvalbatches_y);
    else
        dloss = nneval_batches(dnn, opts, dloss, i, htrainbatches_x, htrainbatches_y);
    end
    
    hloss = cpLossToHost(dloss,opts);

    % plot if figure is available
    if ishandle(fhandle)
        opts.plotfun([], fhandle, hloss, opts, i);
        
        
        %save figure to the output folder after every 10 epochs
        if save_nn_flag && mod(i,10) == 0
            save_figure(fhandle,opts.outputfolder,2,[40 25],14);
            disp(['Saved figure to: ' opts.outputfolder]);
        end
        
    end
    
    t2 = toc(evalt);
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  ...
        '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is: minibatch average: '...
        num2str(mean(L((n-numbatches):(n-1)))) ', epoch error: ' num2str(dloss.train.e(i))])
    disp(['         Eval time: ' num2str(t2) ...
        '. LearningRate: ', num2str(hnn.learningRate) '.Momentum : ' num2str(hnn.momentum)...
        '. free gpu mem (Gb): ', num2str(gpu.FreeMemory./10^9)]);
    
    %save model after every 100 epochs
    if save_nn_flag && mod(i,100) == 0
            epoch_nr = i;
            hnn = cpNNtoHost(dnn);
            save([opts.outputfolder '_epochnr' num2str(epoch_nr) '.mat'],'hnn','opts','epoch_nr','hloss');
            disp(['Saved weights to: ' opts.outputfolder]);
    end

    current_err =  hloss.val.e_errfun(i,:);
    spec_prec_mean = (current_err(2)+current_err(3))/2;
    %save model efter it have the best performance
    if save_nn_flag && (current_err(4) > 0.6 && current_err(5) > 0.75 && spec_prec_mean > 0.6 && spec_prec_mean > spec_prec_mean_old)
            epoch_nr = i;
            spec_prec_mean_old = spec_prec_mean;
            hnn = cpNNtoHost(dnn);
            save([opts.outputfolder  '_best.mat'],'hnn','opts','epoch_nr','hloss');
            disp(['Saved new best weights to: ' opts.outputfolder]);
    end

end

% get network from gpu
hnn = cpNNtoHost(dnn);

%fetch error data from gpu
hloss = cpLossToHost(dloss, opts);

%clear gpu data. nessesary???
clear dnn
clear dbatch_x
clear dbatch_y

reset(gpu);
wait(gpu);
end
