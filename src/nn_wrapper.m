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
% noOfNeuronsPerLayer = [2, 4, 8, 16, 32, length(unique(y))];
noOfNeuronsPerLayer = [size(X, 2), length(unique(y))];
trainRatio = 0.8;
testRatio = 0.1;
epoch = 10;
errThrsd = 0.1;
maxIter = 10000;
eta = [0.1, 0.01, 0.005, 0.001];
% sigmoid, tanh, relu activation functions
% softmax function for last layer; TODO: Not sure if this works
actFnType = {'linear', 'sigmoid', 'tanh', 'relu'};
batchSize = max(1, int16(size(y,1)/10));
% vanillaGD - gradient descent with weight update only after all the training set is feed forward
% vanillaGDRand - same as gradient descent but indices of every batch are randomly generated
% SGD - stochastic gradient descent allows online mini-batch training
solver = {'vanillaGD', 'vanillaGDRand', 'SGD'};

% TODO
% Additional features
% Momentum on weights and learning rate; activation function; 
% network structure;
% K-fold validation

accuracy = zeros(length(actFnType)*length(solver));
valErr = zeros(length(actFnType)*length(solver), epoch);

% nn function
for i=1:length(actFnType)   % Loop for all the activation functions
    for j=1:length(solver) % Loop for all the solvers
        for k=1:length(eta) % Loop for all the learning rates
            disp(['activation : ', actFnType(i),' and solver: ',solver(j), 'learning rate : ', num2str(eta(k))]);
            [accuracy((i-1)*length(solver)*length(eta)+(j-1)*length(eta)+k), valErr((i-1)*length(solver)*length(eta)+(j-1)*length(eta)+k,:)] = nn(X, y, noOfNeuronsPerLayer, trainRatio, testRatio, epoch, errThrsd, maxIter, eta(k), actFnType{i}, batchSize, solver{j});
            disp(['Accuracy is : ', num2str(accuracy((i-1)*length(solver)*length(eta)+(j-1)*length(eta)+k))]);
        end
    end
end

save(['accuracy_.mat'], 'accuracy');
save('valErr_.mat', 'valErr');

end