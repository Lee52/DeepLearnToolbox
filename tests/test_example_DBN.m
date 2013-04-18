
clear *
addpath('../../dataanalysis');
addpath('../data');
addpath('../util');
addpath('../NN');
addpath('../DBN');


load mnist_uint8;
close all
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);



%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(0);
%train dbn
dbn.sizes = [500 500];
opts.numepochs =   25;
opts.batchsize = 100;
opts.momentum  =   0.8;
opts.alpha     =   0.01;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  50;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
figure; visualize(dbn.rbm{1}.W');

assert(er < 0.10, 'Too big error');
