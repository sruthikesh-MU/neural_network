function nn_wrapper(inFile)
% Wrapper function for neural network function

% Load input data and labels
tmp = load(inFile);
y = tmp(:,1);
X = tmp(:,2:end);

% indices = randi(size(tmp, 1), 10, 1);
% y = tmp(indices,1);
% X = tmp(indices,2:end);

% Variance of input data
disp(['Variance of input data for each dimension is :: ',num2str(var(X))]);

% Input parameters
noOfNeuronsPerLayer = [2, 4, 8, 16, 32, length(unique(y))] ;
trainRatio = 0.8;
testRatio = 0.1;
epoch = 100000;
errThrsd = 0.01;
maxIter = 10000;
eta = 0.001;
% sigmoid, tanh, relu activation functions
% softmax function for last layer; TODO: Not sure if this works
actFnType = 'tanh';
batchSize = max(1, int16(size(y,1)/10));
% vanillaGD - gradient descent with weight update only after all the training set is feed forward
% vanillaGDRand - same as gradient descent but indices of every batch are randomly generated
% SGD - stochastic gradient descent allows online mini-batch training
solver = 'SGD';

% TODO
% Additional features
% Momentum on weights and learning rate; activation function; 
% network structure;

% nn function
accuracy = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta, actFnType, batchSize, solver)

end