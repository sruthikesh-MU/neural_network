function nn_wrapper(inFile)
% Wrapper function for neural network function

% Load input data and labels
tmp = load(inFile);
y = tmp(:,1);
X = tmp(:,2:end);

% indices = randi(size(tmp, 1), 10, 1);
% y = tmp(indices,1);
% X = tmp(indices,2:end);


% Input parameters
noOfNeuronsPerLayer = [2, 3];
trainRatio = 0.8;
testRatio = 0.1;
epoch = 128;
errThrsd = 0.01;
maxIter = 10000;
eta = 0.01;
actFnType = 'sigmoid';
batchSize = max(1, int16(size(y,1)/10));

% nn function
accuracy = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta, actFnType, batchSize)

end