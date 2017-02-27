function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

minErrRate = 1;
opt_C = 0;
opt_sigma = 0;
C_candidate = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_candidate =  [0.01,0.03,0.1,0.3,1,3,10,30];
% choose a pair of (C,sigma) to examine the error
for i=1:length(C_candidate)
    for j=1:length(sigma_candidate)
        model= svmTrain(X, y, C_candidate(i), @(x1, x2) gaussianKernel(x1, x2, sigma_candidate(j)));
        predictions = svmPredict(model, Xval);
        errRate = mean(double(predictions ~= yval));
        if errRate < minErrRate
            minErrRate = errRate;
            opt_C = C_candidate(i);
            opt_sigma = sigma_candidate(j);
        end
    end
end
C = opt_C;
sigma = opt_sigma;
% =========================================================================

end
