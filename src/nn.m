function accuracy = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta, actFnType, batchSize)
% Implementation of a simple neural network with back propagation

% Divide input data into train, validation and test set
[trainInd,valInd,testInd] = dividerand(size(X, 1), trainRatio, (1-trainRatio-testRatio), testRatio);    % Indices for input data split
trainX = X(trainInd,:); trainY = y(trainInd); % Training input and labels
valX = X(valInd,:); valY = y(valInd); % Validation input and labels
testX = X(testInd,:); testY = y(testInd); % test input and labels

clearvars trainInd valInd testInd;  % Clear variables, not required anymore

% Data size
[nData, nDim] = size(trainX);

% Data augmentation; Note: Not sure what todo for non-image data augmentation

% Data normalization across dimensions
% zscore normalization
% Note: why not on whole input data? It is better if we just normalize
% training set. may be because we don't want to touch the test with train. Not sure to add validation set.
trainX = zscore(trainX);
valX = zscore(valX);
testX = zscore(testX);

% Initialize input weights
W = init_weights(nDim, noOfNeuronsPerLayer, 0);

% One hot representation of labels
oneHotTrainY = oneHotEncoding(trainY);
oneHotValY = oneHotEncoding(valY);

% Train the network on training set and check the error on validation set
% to update the hyper parameters
W = train(trainX, valX, oneHotTrainY, oneHotValY, noOfNeuronsPerLayer, W, epoch, errThrsd, eta, actFnType, batchSize);

% Deploy the model on the test set to check the accuracy
[labels, accuracy] = test(testX, testY, noOfNeuronsPerLayer, W, actFnType);

end

% Weight Initialization; TODO: Initialization type
function out = init_weights(inpDim, noOfNeuronsPerLayer, initType)

out = cell(length(noOfNeuronsPerLayer), 1);
out{length(noOfNeuronsPerLayer), 1} = [];
for i = 1:length(noOfNeuronsPerLayer)
    if i==1 % First layer, input dimensions will be the dimensions of weights
        out{i} = rand(noOfNeuronsPerLayer(i), inpDim+1) / sqrt(inpDim);
    else    % Other layers, no. of neurons in the previous layer will be the dimensions of current layer weights
        out{i} = rand(noOfNeuronsPerLayer(i), noOfNeuronsPerLayer(i-1)+1) / sqrt(noOfNeuronsPerLayer(i-1));
    end
end
end

% One Hot Encoding
function oneHotLabels = oneHotEncoding(labels)

valueLabels = unique(labels);   % Unique labels which could may not be in sequence
nLabels = length(valueLabels);  % Number of labels
nSamples = length(labels);  % Number of data samples

oneHotLabels = zeros(nSamples, nLabels);
for i = 1:nLabels
    oneHotLabels(:,i) = (labels == valueLabels(i));
end
end

% Train the neural network
function W = train(trainX, valX, trainY, valY, noOfNeuronsPerLayer, W, epoch, errThrsd, eta, actFnType, batchSize)

[nData, nDim] = size(trainX);

% Preallocate cell array
deltaBatch = cell(length(noOfNeuronsPerLayer)+1, 1);
deltaBatch{length(noOfNeuronsPerLayer)+1, 1} = [];
deltaW = cell(length(noOfNeuronsPerLayer)+1, 1);
deltaW{length(noOfNeuronsPerLayer)+1, 1} = [];

% Dropout scheme. If implmented correct the weights to normalize correctly.
epochIter = 0;
ValErr = inf;

% Max no. iterations before terminating the training
while epochIter<epoch && ValErr>errThrsd
    for i = 1:batchSize:nData   % Online training/ batch training
        if(batchSize==1)    % Online training
            randIndices = i;
        else    % Batch training
            randIndices = randi(nData, batchSize, 1); % Generate random indices
        end
        
        % Sample labels for forward pass
        y = trainY(randIndices,:);
        
        % Forward pass
        X=forward(noOfNeuronsPerLayer, actFnType, trainX(randIndices,:), W);  % Feed forward phase of network
        
        % Compute the error
        err = computeErr(X{end}, y);
        % Backward pass
        if batchSize==1 || i==1 % Online training or first batch of training data
            deltaW = backward(actFnType, noOfNeuronsPerLayer, err, X, W);
        else % accumulate all delta
            deltaBatch = backward(actFnType, noOfNeuronsPerLayer, err, X, W);
            for k = 1:length(noOfNeuronsPerLayer) % Accumulate delta over all training samples
                deltaW{k} = deltaW{k} + deltaBatch{k};
            end
        end
        
        % update weights, Overfitting: learning rate momentum, weight decay??
        % TODO: momentum for learning rate
        if batchSize==1
            W = updateWeights(noOfNeuronsPerLayer, W, deltaW, eta); % Online training
        end
        
    end
    % Batch training
    if batchSize~=1
        for k = 1:length(noOfNeuronsPerLayer) % Accumulate delta over all training samples
            deltaW{k} = deltaW{k}./nData;
        end
        W = updateWeights(noOfNeuronsPerLayer, W, deltaW, eta); % Batch training
    end
    
    % Compute the error on validation set and decide which
    % parameters are optimal and check for overfitting and termination
    X=forward(noOfNeuronsPerLayer, actFnType, valX, W);  % Feed forward phase of network
    % Compute the error
    ValErr = sum(sum(abs(computeErr(X{end}, valY))));
    disp(['Iteration: ', num2str(epochIter), ' Validation error: ', ValErr]);
    epochIter = epochIter+1;
end

end

% Test function
function [labels, accuracy] = test(X, y, noOfNeuronsPerLayer, W, actFnType)

% Deploy a forward pass and classify the data
actOut = forward(noOfNeuronsPerLayer, actFnType, X, W);
% Max probability for data classification
[result, labels] = max(actOut{end}, [], 2);
% Compute the error of classification
cp = classperf(y, labels);
accuracy = cp.CorrectRate;
end

% Weight update function
function out = updateWeights(noOfNeuronsPerLayer, W, deltaW, eta)

out = cell(length(noOfNeuronsPerLayer), 1);
out{length(noOfNeuronsPerLayer), 1} = [];

for i=1:length(noOfNeuronsPerLayer)
    out{i} = W{i} - eta.*deltaW{i};
end
end

% Forward pass for the activation function
function out = forward(noOfNeuronsPerLayer, actFnType, X, W)

a = cell(length(noOfNeuronsPerLayer)+1, 1);
a{length(noOfNeuronsPerLayer)+1, 1} = [];

% First layer activations are inputs to the network
a{1} = X;

for i=1:length(noOfNeuronsPerLayer)
    % Local induced field
    v = horzcat(a{i}, ones(size(a{i}, 1), 1))*W{i}';
    
    % activation functions
    a{i+1} = actFn(actFnType, v);
end

out = a;    % Output all the activation of inputs for backpropagation
end

% Activation function
function out = actFn(actFnType, in)

thrsd = 0; % Naive activation function

switch actFnType
    case 'sigmoid'  % Sigmoid activation
        out = 1./(1 + exp(-in));
    case 'tanh' % tanH activation
        out = (2./(1 + exp(-2.*in))) -1;
    case 'relu' % RELU activation, highly sparse
        out = max(0, in);
    case 'softmax'
        out = exp(v)/sum(exp(v));   % probabilities
    otherwise
        out = v>thrsd;  % Heaveside step function
end
end

% Backward pass for the activation function
function out = backward(actFnType, noOfNeuronsPerLayer, err, a, W)

delta = err.*actFnDer(actFnType, a{end}); %Last layer local gradient
for i=length(noOfNeuronsPerLayer):-1:1
    out{i} = (horzcat(a{i}, ones(size(a{i}, 1), 1))'*delta)';
    if i>1 % Local gradient not required for first layer
        delta = actFnDer(actFnType, a{i}) .* (delta*W{i}(:,1:end-1));    % Local gradient for prev layer
    end
end
end

% Derivative of activation function required for backward pass
function out = actFnDer(actFnType, v)

% Derivative of activation function
switch actFnType
    case 'sigmoid'  % Sigmoid activation
        out = v.*(1-v);   % Derivative
    case 'tanh' % tanH activation
        out = 1 - v.^2;
    case 'relu' % RELU activation, highly sparse
        out = v>=0;
    case 'softmax' % Softmax activation function
        out = repmat(v.*(1-v), length(v), 2)*eye(length(v)) + (v*v')*(ones(length(v))-eye(length(v)));   % TODO: Not sure if this is correct
    otherwise
        out = v;    % Heaveside step function
end
end

% Compute error
function out = computeErr(y, trueY)
% Compute the error
out = -(trueY - y); % Derivative of error wrt current neuron y
end
