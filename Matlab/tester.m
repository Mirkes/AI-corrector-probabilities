% tester
% For all tests the QSAR database is used
% Mansouri, K.; Ringsted, T.; Ballabio, D.; Todeschini, R.; Consonni, V.
% Quantitative structure–activity relationship models for ready
% biodegradability of chemicals. J. Chem. Inf. Model. 2013, 53, 867–878.  
% QSAR biodegradation. UCI Machine Learning Repository. Available online:
% https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation (accessed on
% 31 March 2024).  

%% load data and split them to two clases
tab = readtable("qsar.csv");
data = table2array( tab(:, 1:41));
outcome = table2array( tab(:, 42));

%% test of oneDClass and Fisher
% Index of positive (bad) outcome
ind = outcome == 1;
x = data(ind, :);
y = data(~ind, :);
accs = ["BA", "Acc", "f1", "NPV", "PPV", "TPR", "TNR"];
for ac = accs
    [bestT, bestErr, dir] = fisher(x, y, {"LFD score for Binary GOSE outcome", "Bad", "Good"}, ac);
    saveFigures(['figures/', char(ac), '.png']);
end

%% Test specDir
[bestT, bestErr, dir] = fisher(x, y, {"LFD score for Binary GOSE outcome", "Bad", "Good"}, 'ba');
for k = 1:5
    ddir = rand(size(x, 2), 1);
    [bestT, bestErr, dir1] = specDir(x, y, ddir, {"Random score for Binary GOSE outcome", "Bad", "Good"}, 'ba');
    saveFigures(['figures/Random_', num2str(k), '.png']);
end

%% Test of function max search methods.
nRep = 10000;
Da = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5];
Ns = [10, 100, 1000, 10, 100, 1000];
% Grid search
for K = 1:6
    tic;
    for k=1:nRep
        N=10000;
        Delta=Da(K);
        % interval in which we want to find inf / sup of rho and psi
        eps = linspace(0,Delta,N);
        % probability threshold
        N_sample = Ns(K);

        buff=(Delta - eps) .* (1 - 2 * exp(-2 * eps .^2 * N_sample));
        rho = max(max(buff), 0);
    end
    a = toc;
    fprintf("Delta %f Ns %d rho %f time %f\n", Da(K), Ns(K), rho, a);
end
% Dichotomy like
% for K = 1:6
%     tic;
%     Delta=Da(K);
%     N_sample = Ns(K);
%     for k=1:nRep
%         [~, rho] = fminbnd(@(eps) -(Delta - eps) * (1 - 2 * exp(-2 * eps ^2 * N_sample)), 0, Delta);
%     end
%     a = toc;
%     fprintf("Delta %f Ns %d rho %f time %f\n", Da(K), Ns(K), rho, a);
% end

%% Split data to test and training sets
ind = randsample(size(x, 1), floor(0.2 * size(x, 1)));
xTe = x(ind, :);
x(ind, :) = [];
xTr = x;
ind = randsample(size(y, 1), floor(0.2 * size(y, 1)));
yTe = y(ind, :);
y(ind, :) = [];
yTr = y;

% Form model for training set
[bestT, bestErr, dir] = fisher(xTr, yTr, [], 'ba');

% Assess this model for test set
xSc = xTe * dir;
ySc = yTe * dir;

% Form labels and predictions
labX = zeros(length(xSc), 1);
labY = ones(length(ySc), 1);
predX = labX;
predY = labY;
% Form sets for estimation
ind = xSc >= bestT;
predX(ind) = 1;
ind = ySc < bestT;
predY(ind) = 0;

data = [xTe; yTe];
lab = [labX; labY];
pred = [predX; predY];
%
lab = lab+1;
pred = pred+1;

% Test itself
mdl = modelCreator(data, lab, pred);

%% Test of estimate
[gen, det] = estimate(mdl);

%% Test with individual estimations
[gen, det] = estimate(mdl, 'Auto', data, lab);

