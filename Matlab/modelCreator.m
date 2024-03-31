function model = modelCreator(x, labels, prediction, varargin)
%Estimated probabilities of correct acceptance and correct rejection of
%classifier prediction for specified deltas, for Fisher's thresholds or for
%individual score of each case
%
%Inputs
%   x is N-by-M matrix which contains test set points. One row contains one
%       observation.
%   labels is N-by-1 vector with true labels of cases in x. Classes must be
%       integer numbers 1, 2, ..., C, where C is the number of classes. All
%       classes must be presented and for each class must be at least one
%       case correctly predicted and at least one case wrongly predicted.
%   prediction is N-by-1 vector with labels of cases in x, predicted by
%       corrected classifier. Classes must be integer numbers 1, 2, ..., C,
%       where C is the number of classes. All classes must be presented and
%       for each class must be at least one case correctly predicted and at
%       least one case wrongly predicted.
%   Name, Value pairs can include following elements:
%       'name' means list of labels for figures. Default value "None" and
%           assumed absence of figures. Value can be single string:
%           'None' for omitting figures.
%           'Auto' for automatic generation of names.
%           Value also can be array of three strings or cell array of three
%           strings. In this case meaning of strings is 
%           name(1) is name of attribute (title of histogram). It will be
%               used in title of the x-axis and in title of figure. In
%               title of figure to name(1) will be added fragment "for
%               class N" where N will be number of class under
%               consideration.
%           name(2) is name of the first class (the first element of
%               legend) 
%           name(3) is name of the second class (the second element of
%               legend)
%       'Acc' or 'Accuracy' has Value equal to string with name of one of
%           proposed measure of quality to select optimal threshold:
%           'BA' means balanced accuracy: 
%               (TP / (TP + FN) + NT / (TN + FP)) / 2
%           'accuracy' means (TP + TN) / (TP + FN + TN + FP)
%           'f1' means F1 score 2 * TP / (2 * TP + FN + FP)
%           'NPV' means Negative predictive value TN / (TN + FN)
%           'PPV' means Positive predictive value TP / (TP + FP)
%           'TPR', 'recall', 'sens', 'power' means true positive rate
%               (recall, probability of detection, hit rate, power) 
%               TP / (TP + FN)
%           'TNR', 'spec', 'sel', means true positive rate (specificity,
%               selectivity) TN / (TN + FP)
%           Alternatively Value can be function handle. Function must have
%               following syntaxis:  
%                   function acc = funcName(TP, FP, TN, FN) 
%               where TP means true positive, FP means false positive,
%               TN means true negative, FN means false negative.
%       'Dir' or 'Directions has value of string or matrix with directions
%           defined for classes. For string only value 'Fisher' is
%           acceptable. If set of directions is defined than it must be
%           matrix M-by-C (number of rows corresponds to dimension of data
%           space and number of columns corresponds to number of classes).
%           Each column in this matrix corresponds to vector to project
%           sets S-k and S+k to calculate scores, CDF and probabilities.
%
%Outputs
%   model is estimProbCorrector model:
%       sp is cell array with array of scores for set sp for each class
%       sm is cell array with array of scores for set sm for each class
%       thresholds is array with thresholds optimal for each class
%       direct is M-by-C matrix with used direction for each class in
%           corresponding column. If vectors were specified bu user they
%           will be rescaled to unit length and can be inverted to provide
%           low values for rejection.
%       error is array with error estimated for each class for optimal
%           threshold.

    % Test of inputs for correctness
    if ~ismatrix(x) || ~isnumeric(x)
        error(['The first argument x must be real number matrix with',...
            ' one observation in each row.']);
    end
    [N, M] = size(x);
    tmp = false;
    if ~isvector(labels) || ~isvector(prediction) || ~isnumeric(labels)...
            || ~isnumeric(prediction) || length(labels) ~= N ...
            || length(prediction) ~= N || any(floor(labels) ~= labels) ...
            || any(floor(prediction) ~= prediction)
        tmp = true;
    else
        % Check that all elements are presented in labels and prediction
        labels = labels(:);
        prediction = prediction(:);
        [itemsL, ~, ic] = unique(labels);
        countsL = accumarray(ic, 1);    
        [itemsP, ~, ic] = unique(prediction);
        countsP = accumarray(ic, 1);    
        C = size(itemsL, 1);
        if min(countsL) ==0 || min(countsP) == 0 ...
            || C ~= size(itemsL, 1) || C ~= size(itemsP, 1)...
            || min(itemsL) < 1 || max(itemsL) > C ... 
            || min(itemsP) < 1 || max(itemsP) > C
            tmp = true;
        end
    end
    if tmp
        error(['The second and third arguments (labels and prediction)',...
            'must be vectors with number of elements equal to number',...
            ' of rows in matrix x. Elements must be integer numbers',...
            ' 1, 2, ..., C, where C is the number of classes. All',...
            ' classes must be presented and for each class must be',...
            ' at least one case correctly predicted and at least one',...
            ' case wrongly predicted.']);
    end

    % Defining of optional parameters
    name = 'auto';
    acc = 'ba';
    direct = 'fisher';

    % Extraction of optimal parameters
    k = 4;
    while (k < nargin)
        tmp = varargin{k};
        value = varargin{k+1};
        k = k + 2;
        if ~ischar(tmp) || ~isstring(tmp)
            error(['Wrong type of argument %d: %s. Name element of',...
                ' name value pair must be string or char'], k, string(tmp));
        end
        tmp = char(tmp);
        tmp = tmp(1:3);
        if strcmpi(tmp, 'nam')
            name = value;
        elseif strcmpi(tmp, 'acc')
            acc = value;
        elseif strcmpi(tmp, 'dir')
            direct = value;
        else
            error(['Wrong type of argument %d: %s. Name element of',...
                ' name value pair must be string or char'], k, string(tmp));
        end
    end

    % Check optional attributes used by this function
    if ischar(name) || isstring(name)
        if strcmpi(name, 'none')
            name = [];
        elseif strcmpi(name, 'auto')
            name = {"1D projection of score for ", "Rejected", "Accepted"};
        else
            error("Wrong value for name value pair 'name'.")
        end
    end
    % direct
    if ischar(direct) || isstring(direct)
        useFisher = true;
        if ~strcmpi(direct, 'fisher')
            error("Inacceptable string value for argument 'Dir'");
        end
        direct = zeros(M, C);
    else
        useFisher = false;
        if ~ismatrix(direct) || ~isnumeric(direct)... 
           || size(direct, 1) ~= M || size(direct, 2) ~= C
            error(['Matirix "Dir" must have the same number of rows as',...
                ' number of columns in matrix x and the same number of',...
                ' columns as number of classes specified in labels',...
                ' and prediction arguments']);
        end
    end

    % Create object for model.
    model = struct();
    model.sp = cell(C, 1);
    model.sm = cell(C, 1);
    model.thresholds = zeros(C, 1);
    model.direct = direct;
    model.error =  zeros(C, 1);

    % The first round - calculate directions, thresholds, errors and so on.
    % Also fix list of scores for all sp and sm
    for k = 1:C
        % Go throught classes one by one
        % Form sets Sp and Sm:
        ind = prediction == k;
        ind1 = labels == k;
        % sp is set of cases correctly recognised as class k
        sp = x(ind & ind1, :);
        % sm is set of cases wrongly recognised as class k
        sm = x(ind & ~ind1, :);
        name1 = name;
        if ~isempty(name)
            name1{1} = strcat(name{1}, " for class ", num2str(k));
        end
        if useFisher
            % Applied Fisher to find directions and thresholds.
            [bestT, bestErr, dir] = fisher(sm, sp, name1, acc);
        else
            % Directions are specified by user
            [bestT, bestErr, dir] = specDir(sm, sp, direct(:, k), name1, acc);
        end
        model.direct(:, k) = dir;
        model.thresholds(k) = bestT;
        model.error(k) = bestErr;
        model.sp{k} = sort(sp * dir);
        model.sm{k} = sort(sm * dir);
    end
end