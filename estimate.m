function [general, detailed] = estimate(mdl, delta, data, predicted)
% This function is using the previously created model to estimate
% probabilities to correctly reject classifier’s false positive and
% probability to accept classifier’s true positive. Both probabilities are
% estimated from both sides.   
%
% This estimation can be produced for arbitrary specified rejection
% threshold Delta(k), for rejection threshold corresponding to optimal
% thresholds, defined during model creation, or individually for each
% point. The last can be used for estimation of probability that prediction
% is correct.    
%
% This work implemented theory developed in 
% Tyukin, I.Y., Tyukina, T., van Helden, D., Zhang, Z., Mirkes, E.M.,
% Sutton, O.J., Zhou, Q., Gorban, A.N. and Allison, P., 2024. Weakly
% Supervised Learners for Correction of AI Errors with Provable Performance
% Guarantees. arXiv preprint arXiv:2402.00899. 
%
% Inputs:
%   mdl is model created by modelCreator function. THis model contains
%       following fields
%       sp is cell array with array of scores for set sp for each class
%       sm is cell array with array of scores for set sm for each class
%       thresholds is array with thresholds optimal for each class
%       direct is M-by-C matrix with used direction for each class in
%           corresponding column. If vectors were specified bu user they
%           will be rescaled to unit length and can be inverted to provide
%           low values for rejection.
%       error is array with error estimated for each class for optimal
%           threshold.
%   delta specifies rejection threshold for all classes and can has 
%       following values 
%       'Auto' for optimal threshold defined during model creation
%        Real number between 0 and 1 to use the same threshold for all
%           classes. In this case for each class threshold defined by
%           finding of threshold which corresponds to cumulative function
%           equal to delta. 
%        C-by-1 vector contains specified values of delta for each class.
%   data (optional) is N-by-M matrix with data to test. M must be equal to
%       number of rows in matrix direct in model mdl.
%   predicted (optional, must be presented together with data) is N-by-1
%       vector of predicted by classifier labels for records in data.
%       Number of elements in predicted must be the same as number of rows
%       in matrix data. 
%
% Outputs:
%   general is table which calculated for anytime and presented generaal
%       information for specified delta and mdl. general is table with
%       following columns: 
%       Class is column with class number
%       Threshold is used threshold for rejection (if score is less than
%           threshold, then prediction of classifier is rejected)
%       Delta is fraction of false positives to reject (corresponds to
%           Threshold). 
%       Delta_A is fraction of true positives to accept (corresponds to
%           Threshold). 
%       LowBoundReject is lower boundary of probability of correct rejection
%           (equation (8) in journal paper).
%       UpBoundReject is upper boundary of probability of correct rejection
%           (equation (8) in journal paper).
%       LowBoundAccept is lower boundary of probability of correct
%           acceptance (equation (7) in journal paper).
%       UpBoundAccept is upper boundary of probability of correct 
%           acceptance (equation (7) in journal paper).
%   detailed is table with individual output for each point in data. Table
%       contains one row for each row in data. Structure of table is the
%       same as for general but with slightly different meanis of columns:
%       Class is column with class number predicted by classifier (from
%           vector predicted)
%       Threshold is used threshold for rejection (if score is less than
%           threshold, then prediction of classifier is rejected).
%           Threshold is equal to score calculated for corrent point.
%       Delta is fraction of false positives with smaller score
%       Delta_A is fraction of true positives with smaller score
%       LowBoundReject is lower boundary of probability of correct rejection
%           (equation (8) in journal paper).
%       UpBoundReject is upper boundary of probability of correct rejection
%           (equation (8) in journal paper).
%       LowBoundAccept is lower boundary of probability of correct
%           acceptance (equation (7) in journal paper).
%       UpBoundAccept is upper boundary of probability of correct 
%           acceptance (equation (7) in journal paper).

    % Test the first argument
    if ~isstruct(mdl)
        error('The first argument must be structure created by modelCreator');
    end
    fields = fieldnames(mdl);
    if ~all(ismember(["sp", "sm", "thresholds", "direct"], fields))
        error('The first argument must be structure created by modelCreator');
    end

    % Get number of classes
    C = size(mdl.direct, 2);

    % Delta
    if nargin < 2
        delta = 'auto';
    end
    if ischar(delta) || isstring(delta)
        % it must be Auto
        if strcmpi(delta, 'auto')
            delta = zeros(C, 1);
            % Now calculate auto delta
            for k = 1:C
                delta(k) = sum(mdl.sm{k} < mdl.thresholds(k)) /...
                    size(mdl.sm{k}, 1);
            end
        else
            error("String value for delta must be 'Auto'");
        end
    else
        % Specified delta
        if ~isnumeric(delta)
            error("Delta must be string 'Auto' or be numeric.");
        end
        if isscalar(delta)
            % Repeat the same value for all classes
            delta = ones(C, 1) * delta;
        else
            delta = delta(:);
            if size(delta, 1) ~= C
                error(strcat("Numeric Delta must be scalar or be",...
                    " a vector with number of elements equal to",...
                    " number of classes."));
            end
        end
        if (min(delta) <= 0) || max(delta) >= 1
            error("Numeric values of delta must be between 0 and 1 exclude 0 and 1.");
        end
    end

    % Now we are ready to calculate the first output which is general
    % estimation without data
    % Create arrays for result
    clas = (1:C)';
    threshold = zeros(C, 1);
    delta_a = threshold;
    LowBoundReject = threshold;
    UpBoundReject = threshold;
    LowBoundAccept = threshold;
    UpBoundAccept = threshold;
    for k = 1:C
        threshold(k) = inverseCDF(mdl.sm{k}, delta(k));
        delta_a(k) = sum(mdl.sp{k} < threshold(k)) /...
                    size(mdl.sp{k}, 1);
        LowBoundReject(k) = rho(delta(k), size(mdl.sm{k}, 1));
        UpBoundReject(k) = psi(delta(k), size(mdl.sm{k}, 1));
        LowBoundAccept(k) = 1 - psi(delta_a(k), size(mdl.sp{k}, 1));
        UpBoundAccept(k) = 1 - rho(delta_a(k), size(mdl.sp{k}, 1));
    end

    general  = table(clas, threshold, delta, delta_a, LowBoundReject,...
        UpBoundReject, LowBoundAccept, UpBoundAccept, 'VariableNames',...
        ["Class", "Threshold", "Delta", "Delta_a", "LowBoundReject",...
        "UpBoundReject", "LowBoundAccept", "UpBoundAccept"]);
    detailed = [];

    if nargin < 4
        return
    end

    % Check correctness of data defining
    [N, M] = size(data);
    if ~ismatrix(data) || ~isnumeric(data) || M ~= size(mdl.direct, 1)
        error(['The third argument data must be real number matrix with',...
            ' one observation in each row. Number of columns bust be',...
            ' the same as number of rows in matrix mdl.direct']);
    end
    if ~isvector(predicted) || ~isnumeric(predicted)...
         || length(predicted) ~= N || any(floor(predicted) ~= predicted)
        error(['The fourth argument predicted',...
            'must be vectors with number of elements equal to number',...
            ' of rows in matrix data. Elements must be integer numbers',...
            ' 1, 2, ..., C, where C is the number of classes.']);
    end

    % Now we are ready to calculate. Firstly created arrays for usage:
    threshold = zeros(N, 1);
    delta = threshold;
    delta_a = threshold;
    LowBoundReject = threshold;
    UpBoundReject = threshold;
    LowBoundAccept = threshold;
    UpBoundAccept = threshold;
    predicted = predicted(:);

    for r = 1:N
        % get predicted class
        k = predicted(r);
        % calculate score
        threshold(r) = data(r, :) * mdl.direct(:, k);
        delta(r) = sum(mdl.sm{k} < threshold(r)) /...
                    size(mdl.sm{k}, 1);
        delta_a(r) = sum(mdl.sp{k} < threshold(r)) /...
                    size(mdl.sp{k}, 1);
        LowBoundReject(r) = rho(delta(r), size(mdl.sm{k}, 1));
        UpBoundReject(r) = psi(delta(r), size(mdl.sm{k}, 1));
        LowBoundAccept(r) = 1 - psi(delta_a(r), size(mdl.sp{k}, 1));
        UpBoundAccept(r) = 1 - rho(delta_a(r), size(mdl.sp{k}, 1));
    end

    detailed  = table(predicted, threshold, delta, delta_a, LowBoundReject,...
        UpBoundReject, LowBoundAccept, UpBoundAccept, 'VariableNames',...
        ["Class", "Threshold", "Delta", "Delta_a", "LowBoundReject",...
        "UpBoundReject", "LowBoundAccept", "UpBoundAccept"]);
end

function res = rho(a, d)
    % Grid size
    N=1000;
    % Interval in which we want to find inf / sup of rho and psi
    eps = linspace(0, a, 1000);
    % Calculate function
    buff = (a - eps) .* (1 - 2 * exp(-2 * eps .^2 * d));
    res = max(max(buff), 0);
end

function res = psi(a, d)
    % Grid size
    N=1000;
    % Interval in which we want to find inf / sup of rho and psi
    eps = linspace(0, a, 1000);
    % Calculate function
    buff = 2 * exp(-2 * eps .^2 * d) + min(1, eps + a);
    res = min(min(buff), 1);
end

function res = inverseCDF(sets, delta)
    k = floor(delta * length(sets));
    if k < 2
        res = sets(1) - 0.001 * abs(sets(1));
    else
        res = (sets(k) + sets(k + 1)) / 2;
    end
end