function [bestT, bestErr, dir] = specDir(x, y, dir, names, acc)
%Project all data points onto specified direction dir, change direction to
%opposite if necessary to provide less scores for the class 1, normalise
%direction to unit length, select the optimal threshold  
%
%Inputs
%   x is matrix which contains points for Class 1
%   y is matrix which contains points for Class 2
%   dir is direction to project data
%   name (optional) is array of strings or cell array of strings:
%       name(1) is name of attribute (title of histogram)
%       name(2) is name of the first class (the first element of legend)
%       name(3) is name of the second class (the second element of legend)
%       if name is omitted or specified as [] then graphs will not formed.
%   acc (optional) is string or handle of accuracy measure function. For
%   string there are several appropriate values: 
%       'BA' means balanced accuracy: (TP / (TP + FN) + NT / (TN + FP)) / 2
%       'accuracy' means (TP + TN) / (TP + FN + TN + FP)
%       'f1' means F1 score 2 * TP / (2 * TP + FN + FP)
%       'NPV' means Negative predictive value TN / (TN + FN)
%       'PPV' means Positive predictive value TP / (TP + FP)
%       'TPR', 'recall', 'sens', 'power' means true positive rate (recall,
%           probability of detection, hit rate, power) TP / (TP + FN)
%       'TNR', 'spec', 'sel', means true positive rate (specificity,
%           selectivity) TN / (TN + FP)
%       For function handle default value is @balancedAccuracy. function
%       must have following syntaxis: 
%           function acc = funcName(TP, FP, TN, FN) 
%       where TP means true positive, FP means false positive, TN means
%       true negative, FN means false negative.
%
%Outputs
%   bestT is optimal threshold
%   bestErr is minimal error which corresponds to threshold bestT. Error is
%       one minus accuracy defined by acc 
%   dir is vector with used direction
%
    % Direction normalisation
    dir = dir(:) / sqrt(sum(dir .^ 2));

    % Calculate projection
    projX = x * dir;
    projY = y * dir;

    % Check direction and change to opposite if necessary
    if mean(projX) > mean(projY)
        projX = -projX;
        projY = -projY;
        dir = -dir;
    end

    %Calculate threshold and return result
    if nargin < 4
        names = [];
    end
    if nargin < 5
        acc = 'BA';
    end
    [bestT, bestErr] = oneDClass(projX, projY, names, acc);
end