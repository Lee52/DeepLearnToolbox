%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
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
[train_x, mu, sigma]    = zscore(train_x);
test_x                  = normalize(test_x, mu, sigma);

layers = [100 100 100];
rng(0);
sae = saesetup([784 layers]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{1}.weightMaxL2norm          = 15;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
sae.ae{1}.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
sae.ae{1}.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
sae.ae{1}.errfun                   = @nntest;


sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;
sae.ae{2}.weightMaxL2norm          = 10;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
sae.ae{2}.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
sae.ae{2}.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
sae.ae{2}.errfun                   = @nntest;


sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.5;
sae.ae{3}.weightMaxL2norm          = 10;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
sae.ae{3}.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
sae.ae{3}.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
sae.ae{3}.errfun                   = @nntest;


opts.plotfun = @nnupdatefigures;
opts.numepochs = 100;
opts.batchsize = 100;
%opts.momentum_variable      = [linspace(0.5,0.95,1500 ) linspace(0.95,0.95,opts.numepochs -1500)];
%opts.learningRate_variable  =  8.*(linspace(0.998,0.998,opts.numepochs ).^linspace(1,opts.numepochs,opts.numepochs ));
%opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.learningRate_variable  =  (linspace(10,0.2,opts.numepochs ));
opts.momentum_variable      = [linspace(0.5,0.8,3 ) linspace(0.8,0.8,2)];


opts.plot                   = 1;            % 0 = no plotting, migth speed up calc if epochs run fast
opts.ntrainforeval          = 5000;         % number of training samples that are copied to the gpu and used to evalute training performance
                                        
sae = saetrain(sae, train_x, opts);



%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{3}.W{1};
nn.b{1} = sae.ae{1}.b{1};
nn.b{2} = sae.ae{2}.b{1};
nn.b{3} = sae.ae{3}.b{1};


rng(0);
nn                          = nnsetup([784 layers 10]);
nn.output                   = 'softmax'; % output function: softmax | sigm | linear
nn.activation_function      = 'sigm';    % activation func: sigm | tanh_opt | linear
nn.dropoutFraction          = 0.5;       % Droupout of hidden layers
nn.inputZeroMaskedFraction  = 0.2;       % input dropout
%nn.weightPenaltyL2         = 1e-6;      % weightdecay
%nn.weightMaxL2norm          = 15;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
nn.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
nn.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
nn.errfun                   = @nntest;

opts.plotfun                = @nnplottest;
opts.numepochs              =  5000;        %  Number of full sweeps through data
opts.momentum_variable      = linspace(0.5,0.8,opts.numepochs);
opts.learningRate_variable  =  1*(linspace(0.998,0.1,opts.numepochs ));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;            % 0 = no plotting, migth speed up calc if epochs run fast
opts.batchsize              = 100;         % Take a mean gradient step over this many samples. GPU note: below 500 is slow on GPU because of memory transfer
opts.ntrainforeval          = 5000;         % number of training samples that are copied to the gpu and used to evalute training performance
opts.outputfolder           = 'nns/sae'; % saves network each 100 epochs and figures after 10. hinton is prefix to the files. 
                                            % nns is the name of a folder
                                            % from where this script is
                                            % called (probably tests/nns)
                                        
tt = tic;
[nn,L,loss]                 = nntrain(nn, train_x, train_y, opts,test_x,test_y); %use nntrain to train on cpu
toc(tt);
[er_gpu, bad]               = nntest(nn, test_x, test_y);    
fprintf('Error: %f \n',er_gpu);



