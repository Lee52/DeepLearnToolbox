
addpath('../data');
addpath('../util');
addpath('../NN');
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

%% ex1 vanilla neural net
rng(0);
nn                          = nnsetup([784 11 12 10]);
nn.output                   = 'softmax';
nn.activation_function      = {'sigm' 'maxout'};
nn.maxout_pieces            = 3;

nn.W{2} = normrnd(0,0.01, nn.size(2), nn.maxout_pieces, nn.size(3));
nn.b{2} = normrnd(0,0.01, nn.maxout_pieces, nn.size(3));

nn.normalize_input          = 0;
nn.dropoutFraction          = 0.5;
nn.inputZeroMaskedFraction  = 0.2;
nn.weightPenaltyL2          = 1e-6;
%nn.weightMaxL2norm = 15;
nn.cast                     = @double;
nn.caststr                  = 'double';
opts.numepochs              = 10;   %  Number of full sweeps through data
opts.momentum_variable      = [linspace(0.5,0.99,opts.numepochs/2) linspace(0.99,0.99,opts.numepochs/2)];
opts.learningRate_variable  = 2.*(linspace(0.998,0.998,opts.numepochs).^linspace(1,998,opts.numepochs));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
opts.ntrainforeval = 5000; % number of training samples that are copied to the gpu and used to 
                           % evalute training performance
                           % if you have a small dataset set this to number
                           % of samples in your training data
tt = tic;
                           [nn_gpu,L,loss] = nntrain(nn, train_x, train_y, opts);
toc(tt);
                           %[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);
[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
%[er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
fprintf('Error GPU (single): %f \n',er_gpu);
%fprintf('Error GPU (single); %f \n',er_cpu);
