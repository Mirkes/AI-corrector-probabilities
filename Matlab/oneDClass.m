function [bestT, bestErr] = oneDClass(x, y, name, acc)
% oneDClass applied classification with one input attribute by searching
% the best threshold.
%Inputs
%   x contains values for Class 1
%   y contains values for Class 2
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
%       one minus accuracy defined by paramater acc.
%
    % Check definition of all arguments
    if nargin < 3
        name = [];
    end
    if nargin < 4
        acc = 'BA';
    end
    
    % Convert string definition of acc to function handle
    if ischar(acc) || isstring (acc)
        acc = [char(acc), '  '];
        acc = acc(1:3);
        if strcmpi(acc, 'ba ')
            acc = @balancedAccuracy;
        elseif strcmpi(acc, 'acc')
            acc = @accuracy;
        elseif strcmpi(acc, 'f1 ')
            acc = @f1;
        elseif strcmpi(acc, 'npv')
            acc = @NPV;
        elseif strcmpi(acc, 'ppv')
            acc = @PPV;
        elseif strcmpi(acc, 'tpr') || strcmpi(acc, 'rec') ||...
            strcmpi(acc, 'sen') || strcmpi(acc, 'pow')
            acc = @recall;
        elseif strcmpi(acc, 'tnr') || strcmpi(acc, 'sel') ||...
            strcmpi(acc, 'spe') 
            acc = @TNR;
        else
            error("Inacceptable value of requested accuracy. See function oneDClass");
        end
    end

    % Define numbers of cases
    Pos = length(x);
    Neg = length(y);
    tot = Pos + Neg;
    
    % Define set of unique values
    thr = unique([x; y])';
    % Add two boders
    thr = [thr(1) * 0.9999, (thr(2:end) + thr(1:end - 1)) / 2,...
        thr(end) * 1.0001];
    errs = zeros(1, length(thr));
    
    %Define meaning of "class 1"
    xLt =  mean(x) > mean(y);
    
    %Define variabled to search
    bestErr = tot;
    bestT = -Inf;
    %Check each threshold
    for k = 1:length(thr)
        t = thr(k);
        nX = sum(x < t);
        nY = sum(y >= t);
        if xLt
            nX = Pos - nX;
            nY = Neg - nY;
        end
        err = 1 - acc(nX, Neg - nY, nY, Pos - nX);
        if err < bestErr
            bestErr = err;
            bestT = t;
        end
        errs(k) = err;
    end

    if ~isempty(name)
        if ~iscell(name)
            name = cellstr(name);
        end
        
        %Define min and max to form bines
        mi = min([x; y]);
        ma = max([x; y]);
        edges = mi:(ma-mi)/20:ma;
        
        %Draw histograms
        figure;
        histogram(x, edges, 'Normalization','probability');
        hold on;
        histogram(y, edges, 'Normalization','probability');
        title(name{1});
        xlabel(['Value of ', name{1}]);
        ylabel('Fraction of cases');
        
        %Draw graph of errors
        sizes = axis();
        plot(thr, errs * sizes(4), 'g');
        %Draw the best threshold
        plot([bestT, bestT], sizes(3:4), 'k', 'LineWidth', 2);
        legend(name{2}, name{3}, 'Error', 'Threshold');
    end
end

function acc = balancedAccuracy(TP, FP, TN, FN) 
    acc = (TP / (TP + FN) + TN / (TN + FP)) / 2;
end

function acc = accuracy(TP, FP, TN, FN) 
    acc = (TP + TN) / (TP + FN + TN + FP);
end

function acc = f1(TP, FP, ~, FN) 
    acc = 2 * TP / (2 * TP + FN + FP);
end

function acc = NPV(~, ~, TN, FN) 
    acc = TN / (TN + FN);
end

function acc = PPV(TP, FP, ~, ~) 
    acc = TP / (TP + FP);
end

function acc = recall(TP, ~, ~, FN) 
    acc = TP / (TP + FN);
end

function acc = TNR(~, FP, TN, ~) 
    acc = TN / (TN + FP);
end
