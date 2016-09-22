function [accuracy, valErr] = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta, actFnType, batchSize, solver)
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
[W, valErr] = train(trainX, valX, oneHotTrainY, oneHotValY, noOfNeuronsPerLayer, W, epoch, errThrsd, eta, actFnType, batchSize, solver);

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
function [W, valErr] = train(trainX, valX, trainY, valY, noOfNeuronsPerLayer, W, epoch, errThrsd, eta, actFnType, batchSize, solver)

[nData, nDim] = size(trainX);
etaOrig = eta;

% Preallocate cell array
deltaBatch = cell(length(noOfNeuronsPerLayer)+1, 1);
deltaBatch{length(noOfNeuronsPerLayer)+1, 1} = [];
deltaW = cell(length(noOfNeuronsPerLayer)+1, 1);
deltaW{length(noOfNeuronsPerLayer)+1, 1} = [];

% Dropout scheme. If implmented correct the weights to normalize correctly.
epochIter = 0;
valErr = inf;
randIndices = 1:nData;

% Max no. iterations before terminating the training
while epochIter<epoch && valErr>errThrsd
    
    % Shuffle the indices every iteration to have a differnet update
    randIndices = (randperm(nData))';
    
    for i = 1:batchSize:nData   % Online training/ batch training
        
        curBatch = i/batchSize + 1;
        
        % Generate batch indices completely random; Somehow this gives a
        % good accuracy because this way input is covered randomly
        if strcmp(solver, 'vanillaGDRand')
            if curBatch*batchSize > nData
                randIndices((curBatch-1)*batchSize+1:nData) = randi(nData, rem(nData, batchSize), 1); % Generate random indices
            else
                randIndices((curBatch-1)*batchSize+1:curBatch*batchSize) = randi(nData, batchSize, 1); % Generate random indices
            end
        end
        
        % Sample labels for forward pass
        if curBatch*batchSize > nData
            y = trainY(randIndices((curBatch-1)*batchSize+1:nData),:);
            % Forward pass
            X=forward(noOfNeuronsPerLayer, actFnType, trainX(randIndices((curBatch-1)*batchSize+1:nData),:), W);  % Feed forward phase of network
        else
            y = trainY(randIndices((curBatch-1)*batchSize+1:curBatch*batchSize),:);
            % Forward pass
            X=forward(noOfNeuronsPerLayer, actFnType, trainX(randIndices((curBatch-1)*batchSize+1:curBatch*batchSize),:), W);  % Feed forward phase of network
        end
        
        % Compute the error
        err = computeErr(X{end}, y);
        % Backward pass
        if strcmp(solver, 'SGD') || curBatch==1 % Online training or first batch of training datai
            deltaW = backward(actFnType, noOfNeuronsPerLayer, err, X, W);
        else % accumulate all delta
            deltaBatch = backward(actFnType, noOfNeuronsPerLayer, err, X, W);
            for k = 1:length(noOfNeuronsPerLayer) % Accumulate delta over all training samples
                deltaW{k} = deltaW{k} + deltaBatch{k};
            end
        end
        % update weights, Overfitting: learning rate momentum, weight decay??
        % TODO: momentum for learning rate
        if strcmp(solver, 'SGD')
            W = updateWeights(noOfNeuronsPerLayer, W, deltaW, eta); % Stochastic gradient online training
        end
        
    end
    % Batch training
    if strcmp(solver, 'vanillaGD') || strcmp(solver, 'vanillaGDRand')
        for k = 1:length(noOfNeuronsPerLayer) % Accumulate delta over all training samples
            deltaW{k} = deltaW{k}./nData;
        end
        W = updateWeights(noOfNeuronsPerLayer, W, deltaW, eta); % Batch training
    end
    
    % Compute the error on validation set and decide which
    % parameters are optimal and check for overfitting and termination
    X=forward(noOfNeuronsPerLayer, actFnType, valX, W);  % Feed forward phase of network
    % Compute the error
    valErr = sum(sum(abs(computeErr(X{end}, valY))));
    if mod(epochIter, 1000)==0
        disp(['Iteration: ', num2str(epochIter), ' Validation error: ', num2str(valErr)]);
    end
    
    % Annealing learning rate
    % Multiple ways to do this; using the gradient difference or using
    % iteration number; 
    % For simplicity we use iteration count
    k = 1e-4;
    eta = etaOrig*exp(-k*epochIter);    % exponential decay: CS231n
    
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

% TODO: Other solver optimizations 
% Momentum, Nesterov, Adagrad, Adadelta, Adam

% Vanilla/ Batch Gradient descent
% SGD - Online training for each sample
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
    case 'linear'   % Identity activation
        out = in;   
    case 'sigmoid'  % Sigmoid activation
        out = 1./(1 + exp(-in));
    case 'tanh' % tanH activation
        out = (2./(1 + exp(-2.*in))) -1;
    case 'relu' % RELU activation, highly sparse
        out = max(0, in);
    case 'softmax'
        out = exp(in)/sum(exp(in));   % probabilities
    otherwise
        out = in>thrsd;  % Heaveside step function
end
end

% Backward pass for the activation function
function out = backward(actFnType, noOfNeuronsPerLayer, err, a, W)

delta = err.*actFnDer(actFnType, a{end}); %Last layer local gradient
for i=length(noOfNeuronsPerLayer):-1:1
    out{i} = (horzcat(a{i}, ones(size(a{i}, 1), 1))'*delta)'; % Sum of all the delta for the batch of inputs
    if i>1 % Local gradient not required for first layer
        delta = actFnDer(actFnType, a{i}) .* (delta*W{i}(:,1:end-1));    % Local gradient for prev layer
    end
end
end

% Derivative of activation function required for backward pass
function out = actFnDer(actFnType, v)

% Derivative of activation function
switch actFnType
    case 'linear'   % Identity activation
        out = 1;
    case 'sigmoid'  % Sigmoid activation
        out = v.*(1-v);   % Derivative
    case 'tanh' % tanH activation
        out = 1 - v.^2;
    case 'relu' % RELU activation, highly sparse
        out = v>0;
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
