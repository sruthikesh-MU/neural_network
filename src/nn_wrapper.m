function nn_wrapper(inFile)
% Wrapper function for neural network function

% Load input data and labels
tmp = load(inFile);
y = tmp(:,1);
X = tmp(:,2:end);

% Input parameters
noOfNeuronsPerLayer = [4, 3];
trainRatio = 0.8;
testRatio = 0.1;
epoch = 16;
errThrsd = 0.01;
maxIter = 128;
eta = 0.01;
actFnType = 'sigmoid';
batchSize = size(y,1)/10;

% nn function
out = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta, actFnType, batchSize)

end