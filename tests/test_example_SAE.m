clear *
addpath('../../dataanalysis');
addpath('../data');
addpath('../util');
addpath('../NN');
addpath('../SAE');


load mnist_uint8;
close all
cast = @double;

train_x = cast(train_x) / 255;
test_x  = cast(test_x)  / 255;
train_y = cast(train_y);
test_y  = cast(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

rng(0);
sae = saesetup([784 100 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{1}.output                   = 'softmax';        % output function: softmax | sigm | linear
sae.ae{1}.activation_function      = 'sigm';           % activation func: sigm | tanh_opt | linear
sae.ae{1}.dropoutFraction          = 0.5;              % Droupout of hidden layers
sae.ae{1}.inputZeroMaskedFraction  = 0.2;              % input dropout
%nn.weightPenaltyL2         = 1e-6;             % weightdecay
sae.ae{1}.weightMaxL2norm          = 15;               % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
sae.ae{1}.cast                     = @double;          % double or single precision, single cuts memory usage by app. 50%
sae.ae{1}.caststr                  = 'double';         % double or single precision, single cuts memory usage by app. 50%
sae.ae{1}.errfun                   = @nntest;          % The error function used

visualize(sae.ae{1}.W{1}')

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;
sae.ae{2}.output                   = 'softmax';        % output function: softmax | sigm | linear
sae.ae{2}.activation_function      = 'sigm';           % activation func: sigm | tanh_opt | linear
sae.ae{2}.dropoutFraction          = 0.5;              % Droupout of hidden layers
sae.ae{2}.inputZeroMaskedFraction  = 0.2;              % input dropout
%nn.weightPenaltyL2         = 1e-6;             % weightdecay
sae.ae{2}.weightMaxL2norm          = 15;               % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
sae.ae{2}.cast                     = @double;          % double or single precision, single cuts memory usage by app. 50%
sae.ae{2}.caststr                  = 'double';         % double or single precision, single cuts memory usage by app. 50%
sae.ae{2}.errfun  


opts.numepochs              = 10;            %  Number of full sweeps through data
opts.batchsizeforeval       = 10000;            % the number of evalution samples used after each epoch
opts.maxevalbatches         = 1;                % number of minibathes the evalution data is split into. Increase to lower memory usage
opts.plotfun                = @nnplottest;      % plotting function.
opts.momentum_variable      = [linspace(0.5,0.85,opts.numepochs/2 ) linspace(0.85,0.85,opts.numepochs/2)];
opts.learningRate_variable  =  8.*(linspace(0.998,0.1,opts.numepochs));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;                 % 0 = no plotting, migth speed up calc if epochs run fast
opts.batchsize              = 100;               % Take a mean gradient step over this many samples. GPU note: below 500 is slow on GPU because of memory transfer
opts.outputfolder           = 'nns/hinton';  




sae = saetrain(sae, train_x, opts);


% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
%opts.numepochs =   1;
%opts.batchsize = 100;
%[nn,L,loss] = nntrain(nn, train_x, train_y, opts);
%[er, bad] = nntest(nn, test_x, test_y);
%assert(er < 0.1, 'Too big error');
