%RUN FROM tests directory in DEEPLEARNTOOLBOX (skaae version) folder
batch_x = rand(20, 5);
batch_y = rand(20, 2);
rng(0)
addpath('../NN')
addpath('../util')
nn = nnsetup([5 4 2]);
nn.momentum = 0.5;
nn.inputZeroMaskedFraction = 0.2;
nn.dropoutFraction = 0.5;
nn.weightPenaltyL2 = 0.001;
nn.activation_function = 'tanh_opt';
nn.output = 'softmax';
nn.dropoutFraction = 0;


nnbw = nnff(nn, batch_x, batch_y);
nnbw = nnbp(nnbw);
nnbw = nnapplygrads(nnbw);

rmpath('../NN')
rmpath('../util')

%PATH TO RASMUS BERGS PALMS ORIGINAL DEEP LEARN TOOLBOX
cd '/Users/casperkaae/Documents/Uni/machine_learing_for_signal_processing/DeepLearnToolbox-original/tests'
addpath('../NN')
addpath('../util')
rng(0)
nnorg = nn;
nnorg.W{1} = [nn.b{1} nn.W{1}];
nnorg.W{2} = [nn.b{2} nn.W{2}];
nnorg.vW{1} = [nn.vb{1} nn.vW{1}];
nnorg.vW{2} = [nn.vb{2} nn.vW{2}];
nnorg = nnff(nnorg, batch_x, batch_y);
nnorg = nnbp(nnorg);
nnorg = nnapplygrads(nnorg);

rmpath('../NN')
rmpath('../util')
cd '/Users/casperkaae/Documents/Uni/machine_learing_for_signal_processing/DeepLearnToolbox.git/tests'

e = 1e-28;
assert( e>sumsqr([nnbw.db{1} nnbw.dW{1}] - [nnorg.dW{1}]),'grad1')
assert( e>sumsqr([nnbw.db{2} nnbw.dW{2}] - [nnorg.dW{2}]),'grad2')
assert( e>sumsqr([nnbw.b{1} nnbw.W{1}] - [nnorg.W{1}]),'W1')
assert( e>sumsqr([nnbw.b{2} nnbw.W{2}] - [nnorg.W{2}]),'W2')
assert( e>sumsqr([nnbw.a{1}-nnorg.a{1}(:,2:end)]),'a1')
assert( e>sumsqr([nnbw.a{2}-nnorg.a{2}(:,2:end)]),'a2')
assert( e>sumsqr([nnbw.a{3}-nnorg.a{3}]),'a3')
disp('All Tests Passed');