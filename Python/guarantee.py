# -*- coding: utf-8 -*-
"""
This module includes functions for calculation of guaranteed probability of
correct acceptance/rejection of original classifier decision

This module implemented theory developed in
add reference after the paper acceptance

Main functionality is presented in class rejectAcceptModel. Constructor created
    model and then general or tmpividual estimations can be calculated through
    methods generalEstimate and detailEstimate, correspondingly.

function fisher serves for calculation of Fisher's linear discriminant
    direction (without assumption of normality of distribution or equality
    of covariance matrix in both classes.'

function oneDClass serves to ftmp the optimal threshold for 1D classifier and,
    if requested to draw figure with this threshold

Service functions are used to calculate tmpicator of classfier quality to
    search optimal threshold. Default value is "BA" and corresponds to balanced
    accuracy. List of possible functions is presented below:
        ba calculates balanced accuracy: (TP / (TP + FN) + NT / (TN + FP)) / 2
        accuracy calculates (TP + TN) / (TP + FN + TN + FP)
        f1 calculates F1 score: 2 * TP / (2 * TP + FN + FP)
        npv calculates Negative predictive value TN / (TN + FN)
        ppv calculates Positive predictive value TP / (TP + FP)
        tpr calculates true positive rate  TP / (TP + FN)
        tnr calculates true negative rate  TN / (TN + FP)

Created on Thu Mar  7 18:48:56 2024

@author: em322
"""
# pylint: disable=invalid-name

import numbers
import math
import numpy as np
import matplotlib.pyplot as plt

class RejectAcceptModel:
    '''
    This class creates model of corrector and then can be used to estimate
    probabilities of correct rejection/accepance of classifier decisions
    Class constructor used data about point dataset, pure labels and
    predictions of classifier to create model. There are several ways to
    form model and all these ways are described in constructor.
    '''
    # Attributes of model
    sp = None # is list with array of scores for set sp for each class
    sm = None # is list with array of scores for set sm for each class
    thresholds = None # is array with thresholds optimal for each class
    direct = None # is M-by-C matrix with used direction for each class in
           # corresponding column. If vectors were specified by user they
           # will be rescaled to unit length and can be inverted to provide
           # low values for rejection.
    error = None # is array with error estimated for each class for optimal
           # threshold.

    def __init__(self, x, labels, prediction,\
                 name = None, acc = 'ba', direct = 'Fisher'):
        '''
        It is constructor of class rejectAcceptModel. It used specified
        parameters and used them to create model

        Parameters
        ----------
        x : 2D ndarray
            DESCRIPTION. N-by-M matrix which contains test set points. One row
            contains one observation
        labels : 1D ndarray, list or tuple
            DESCRIPTION. N-by-1 vector with true labels of cases in x. Classes
            must be integer numbers 1, 2, ..., C, where C is the number of
            classes. All classes must be presented and for each class must
            be at least one case correctly predicted and at least one case
            wrongly predicted.
        prediction : 1D ndarray, list or tuple
            DESCRIPTION. is N-by-1 vector with labels of cases in x, predicted
            by corrected classifier. Classes must be integer numbers
            1, 2, ..., C, where C is the number of classes. All classes must
            be presented and for each class must be at least one case correctly
            predicted and at least one case wrongly predicted.
        name : None, string or array of strings, optional
            DESCRIPTION. Possible values and their meaning:
           None for omitting figures.
           'Auto' for automatic generation of names.
           Name also can be array of three strings In this case meaning of
           strings is
           name(1) is name of attribute (title of histogram). It will be
               used in title of the x-axis and in title of figure. In
               title of figure to name(1) will be added fragment "for
               class N" where N will be number of class under
               consideration.
           name(2) is name of the first class (the first element of legend)
           name(3) is name of the second class (the second element of legend)
            The default is None and assumed absence of figures.
        acc : defined in function oneDClass, optional
            DESCRIPTION. defined in function oneDClass. The default is 'ba'.
        direct : string or M-by-C matrix (2D ndarray), optional
            DESCRIPTION. For string only value 'Fisher' is acceptable.
            If set of directions is defined than it must be matrix M-by-C
            (number of rows corresponds to dimension of data space and number
            of columns corresponds to number of classes). Each column in this
            matrix corresponds to vector to project sets S-k and S+k to
            calculate scores, CDF and probabilities
            The default is 'Fisher'.

        Raises
        ------
        RuntimeError
            DESCRIPTION. Raised if argument are inappropriate. Text of
            exception described what should be sent to method.

        Returns
        -------
        Object of class rejectAcceptModel.

        '''
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-branches

        # Test of inputs for correctness
        if  not isinstance(x, np.ndarray) or x.ndim != 2:
            raise RuntimeError('The first argument x must be real number 2D\
                               ndarray with one observation in each row.')
        # Just in case convert labels and prediction and subtract 1 to have 0
        # base classes
        labels = np.asarray(labels, dtype=np.int32) - 1
        prediction = np.asarray(prediction, dtype=np.int32) - 1
        # Check that all elementa are predicted in labels and prediction
        tmp = np.unique(labels)
        C = tmp.shape[0]
        tmp1 = np.unique(prediction)
        if tmp[0] != 0 or tmp[-1] != C-1 or tmp1[0] != 0 or tmp1[-1] != C-1\
            or tmp1.shape[0] != C:
            raise RuntimeError('The second and third arguments' +\
                '(labels and prediction) must be vectors with number of ' +\
                'elements equal to number of rows in matrix x. Elements ' +\
                'must be integer numbers 1, 2, ..., C, where C is the' +\
                'number of classes. All classes must be presented and for ' +\
                'each class must be at least one case correctly predicted ' +\
                'and at least one case wrongly predicted.')

        # Check optional attributes used by this function
        if name is not None:
            if isinstance(name, str):
                if name.lower() == 'auto':
                    name = np.asarray(["1D projection of score for ",\
                                       "Rejected", "Accepted"])
                else:
                    raise RuntimeError("Wrong string value for name argument."\
                        + "It must be 'Auto'")

        # Check value of direct
        if isinstance(direct, str):
            useFisher = True
            if direct.lower() != 'fisher':
                raise RuntimeError("Inacceptable string value for argument 'direct'")
            direct = np.zeros((x.shape[1], C))
        else:
            useFisher = False
            if not isinstance(direct, np.ndarray) or direct.ndim != 2\
                or direct.shape != (x.shape[1], C):
                raise RuntimeError('Matirix "Dir" must have the same number' +\
                    ' of rows as number of columns in matrix x and the same' +\
                    ' number of columns as number of classes specified in' +\
                    ' labels and prediction arguments')

        # Initialise attributes of object
        self.sp = list(range(C))
        self.sm = list(range(C))
        self.thresholds = np.zeros(C)
        self.direct = direct
        self.error = np.zeros(C)
        # The first round - calculate directions, thresholds, errors and so on.
        # Also fix list of scores for all sp and sm
        for k in range(C):
            # Go throught classes one by one
            # Form sets Sp and Sm:
            tmp = prediction == k
            tmp1 = labels == k
            # sp is set of cases correctly recognised as class k
            sp = x[tmp & tmp1, :]
            # sm is set of cases wrongly recognised as class k
            sm = x[tmp & ~tmp1, :]
            name1 = name
            if name is not None:
                name1[0] = name[0] + " for class " , str(k + 1)
            if useFisher:
                # Applied Fisher to ftmp directions and thresholds.
                [self.thresholds[k], self.error[k], self.direct[:, k] ] = \
                    fisher(sm, sp, name1, acc)
            else:
                # Directions are specified by user
                [self.thresholds[k], self.error[k], self.direct[:, k]] = \
                    specDir(sm, sp, direct[:, k], name1, acc)
            self.sp[k] = np.sort(np.matmul(sp, self.direct[:, k]))
            self.sm[k] = np.sort(np.matmul(sm, self.direct[:, k]))

    def generalEstimate(self, delta='Auto'):
        '''
        This function is using the previously created model to estimate
        probabilities to correctly reject classifier’s false positive and
        probability to accept classifier’s true positive. Both probabilities
        are estimated from both sides.
        Probabilities estimated for each class and for specified threshold of
        rejection delta

        Parameters
        ----------
        delta : string or float or vector of float
            DESCRIPTION. delta specifies rejection threshold for all classes
            and can has following values
                'Auto' for optimal threshold defined during model creation
                Real number between 0 and 1 to use the same threshold for all
                    classes. In this case for each class threshold defined by
                    ftmping of threshold which corresponds to cumulative
                    function equal to delta.
                C-by-1 vector with specified values of delta for each class.
                    C is number of classes

        Raises
        ------
        RuntimeError
            DESCRIPTION. Raised if argument are inappropriate. Text of
            exception described what should be sent to method.

        Returns
        -------
        res : 2D ndarray
            DESCRIPTION. This array contains number of rows equal to number of
            cllasses C and 8 columns:
            0. Class is column with class number from 1 to C
            1. Threshold is used threshold for rejection (if score is less than
                threshold, then prediction of classifier is rejected)
            2. Delta is fraction of false positives to reject (corresponds to
                Threshold).
            3. Delta_A is fraction of true positives to accept (corresponds to
                Threshold).
            4. LowBoundReject is lower boundary of probability of correct
                rejection (equation (8) in journal paper).
            5. UpBoundReject is upper boundary of probability of correct
                rejection (equation (8) in journal paper).
            6. LowBoundAccept is lower boundary of probability of correct
                acceptance (equation (7) in journal paper).
            7. UpBoundAccept is upper boundary of probability of correct
                acceptance (equation (7) in journal paper).
        '''
        # Get number of classes
        C = self.error.shape[0]

        # Create matrix for results
        res = np.zeros((C, 8))
        # Put classes
        res[:, 0] = np.asarray(range(C)) + 1

        # Delta
        if isinstance(delta, str):
            # it must be Auto
            if delta.lower() == 'auto':
                # Now calculate auto delta
                for k in range(C):
                    res[k, 2] = sum(self.sm[k] < self.thresholds[k]) / \
                        self.sm[k].shape[0]
            else:
                raise RuntimeError("String value for delta must be 'Auto'")
        else:
            # Specified delta
            if isinstance(delta, numbers.Number):
                #  Repeat the same value for all classes
                res[:, 2] = np.ones(C) * delta
            else:
                #it is array
                res[:, 2] = delta
            # Check correctness of delta
            if (min(res[:, 2]) <= 0) or max(res[:, 2]) >= 1:
                raise RuntimeError("Numeric values of delta must be" \
                                   + " between 0 and 1 exclude 0 and 1.")

        # Now we are ready to calculate the general estimation
        for k in range(C):
            res[k, 1] = inverseCDF(self.sm[k], res[k, 2])
            res[k, 3] = sum(self.sp[k] < res[k, 1]) / \
                        self.sp[k].shape[0]
            res[k, 4] = rho(res[k, 2], self.sm[k].shape[0])
            res[k, 5] = psi(res[k, 2], self.sm[k].shape[0])
            res[k, 6] = 1 - psi(res[k, 3], self.sp[k].shape[0])
            res[k, 7] = 1 - rho(res[k, 3], self.sp[k].shape[0])
        return res

    def detailEstimate(self, data, predicted):
        '''
        This function is using the previously created model to estimate
        probabilities to correctly reject classifier’s false positive and
        probability to accept classifier’s true positive. Both probabilities
        are estimated from both sides.
        Probabilities are estimated for each point (row of matrix data)
        tmpividually

        Parameters
        ----------
        data : 2D ndarray of float
            DESCRIPTION. data is N-by-M matrix with data to test. M must be
            equal to number of rows in matrix direct in object self of class
            rejectAcceptModel.
        predicted : 1D array or array like type
            DESCRIPTION. predicted is N element vector of predicted by
            classifier labels for records in data. Number of elements in
            predicted must be the same as number of rows in matrix data.

        Raises
        ------
        RuntimeError
            DESCRIPTION. Raised if argument are inappropriate. Text of
            exception described what should be sent to method.

        Returns
        -------
        res : TYPE 2D ndarray
            DESCRIPTION. res is matrix with tmpividual output for each point in
            data. Matrix contains one row for each row in data. Meanings of
            columns of matrix:
            Class is column with class number predicted by classifier (from
                vector predicted)
            Threshold is used threshold for rejection (if score is less than
                threshold, then prediction of classifier is rejected).
                Threshold is equal to score calculated for current point.
       Delta is fraction of false positives with smaller score
       Delta_A is fraction of true positives with smaller score
       LowBoundReject is lower boundary of probability of correct rejection
           (equation (8) in journal paper).
       UpBoundReject is upper boundary of probability of correct rejection
           (equation (8) in journal paper).
       LowBoundAccept is lower boundary of probability of correct
           acceptance (equation (7) in journal paper).
       UpBoundAccept is upper boundary of probability of correct
           acceptance (equation (7) in journal paper).


        '''
        # Check correctness of data defining
        if  not isinstance(data, np.ndarray) or data.ndim != 2 \
            or data.shape[1] != self.direct.shape[0]:
            raise RuntimeError('The argument data must be real number'\
                + ' matrix with one observation in each row. Number of'\
                + ' columns must be the same as number of rows in '\
                + 'attribute direct')
        N = data.shape[0]
        try:
            predicted = np.asarray(predicted, dtype=np.int32) - 1
            if predicted.shape[0] != N or \
                np.max(predicted) >= self.error.shape[0] or \
                np.min(predicted) < 0:
                raise RuntimeError()
        except Exception as exc:
            raise RuntimeError('The argument predicted must be vector'\
                + ' with number of elements equal to number of rows in matrix'\
                + ' data. Elements must be integer numbers 1, 2, ..., C,'\
                + ' where C is the number of classes.') from exc
        # Now we are ready to calculate. Firstly created array for usage:
        res = np.zeros((N, 8))
        for r in range(N):
            # Get predicted class
            k = predicted[r]
            res[r, 0] = k + 1
            # Calculate score
            res[r, 1] = np.matmul(data[r, :], self.direct[:, k])
            res[r, 2] = sum(self.sm[k] < res[r, 1]) / \
                        self.sm[k].shape[0]
            res[r, 3] = sum(self.sp[k] < res[r, 1]) / \
                        self.sp[k].shape[0]
            res[r, 4] = rho(res[r, 2], self.sm[k].shape[0])
            res[r, 5] = psi(res[r, 2], self.sm[k].shape[0])
            res[r, 6] = 1 - psi(res[r, 3], self.sp[k].shape[0])
            res[r, 7] = 1 - rho(res[r, 3], self.sp[k].shape[0])
        return res

def inverseCDF(sets, delta):
    '''
    This function calculate pseudo inverse of empirical cumulative density
    function for set sets and for fraction of rejected (smaller than) delta.

    Parameters
    ----------
    sets : 1D ndarray
        DESCRIPTION. Array of sorted scores.
    delta : float
        DESCRIPTION. Fraction or rejected scores (less than calculated
        threshold).

    Returns
    -------
    res : float
        DESCRIPTION. Value of threshold such that fraction or rejected scores
        (less than res) is equal to delta
    '''
    # Position is one less than number of elements
    k = math.floor(delta * sets.shape[0]) - 1
    if k < 1:
        res = sets[0] - 0.001 * abs(sets[0])
    elif k > sets.shape[0] - 2:
        res = sets[-1] + 0.001 * abs(sets[-1])
    else:
        res = (sets[k] + sets[k + 1]) / 2
    return res

def rho(a, d):
    '''
    Calculate function rho - formula (9) of paper

    Parameters
    ----------
    a : float
        DESCRIPTION. Tolerance fraction
    d : int
        DESCRIPTION. Number of elements in corresponding dataset

    Returns
    -------
    float
        DESCRIPTION. probability that decision is correct

    '''
    # Interval in which we want to ftmp inf / sup of rho and psi
    eps = np.linspace(0, a, 1000)
    # Calculate function
    buff = (a - eps) * (1 - 2 * np.exp(-2 * np.square(eps) * d))
    return max(np.max(buff), 0)

def psi(a, d):
    '''
    Calculate function psi - formula (10) of paper

    Parameters
    ----------
    a : float
        DESCRIPTION. Tolerance fraction
    d : int
        DESCRIPTION. Number of elements in corresponding dataset

    Returns
    -------
    float
        DESCRIPTION. probability that decision is correct

    '''
    # Interval in which we want to ftmp inf / sup of rho and psi
    eps = np.linspace(0, 1 - a, 1000)
    # Calculate function
    buff = 2 * np.exp(-2 * np.square(eps) * d) + eps + a
    return min(np.min(buff), 1)

def specDir(x, y, direct, names=None, acc = 'BA'):
    '''
    Project all data points onto specified direction direct, change direction to
    opposite if necessary to provide less scores for the class 1, normalise
    direction to unit length, select the optimal threshold

    Parameters
    ----------
    x : 2D ndarray of floats
        DESCRIPTION. Matrix with data of class 0. Each row contains one
        mobservation, each column contains one variable
    y : 2D ndarray of floats
        DESCRIPTION. Matrix with data of class 1. Each row contains one
        mobservation, each column contains one variable
    direct : 1D ndarray of floats
        DESCRIPTION. dir is direction to project data
    names : defined in function oneDClass
        DESCRIPTION. defined and used in function oneDClass.
    acc : defined in function oneDClass
        DESCRIPTION. defined and used in function oneDClass.

    Returns
    -------
    bestT float
        DESCRIPTION. Optimal threshold value.
    bestErr float
        DESCRIPTION. best Err is minimal error which corresponds to threshold
        bestT. Error is one minus accuracy defined by acc
    direct 1dimensional ndarray of floats
        DESCRIPTION. dir is vector of fisher direction
    '''

    #  Direction normalisation
    direct = direct / np.sqrt(np.sum(np.square(direct)))

    # Calculate projection
    projX = np.matmul(x, direct)
    projY = np.matmul(y, direct)

    # Check direction and change to opposite if necessary
    if np.mean(projX) > np.mean(projY):
        projX = -projX
        projY = -projY
        direct = -direct

    # Calculate threshold and return result
    (bestT, bestErr) = oneDClass(projX, projY, names, acc)
    return bestT, bestErr, direct

def fisher(x, y, names=None, acc = 'BA'):
    '''
    Calculate Fisher's discriminant direction, project all data points onto
    this direction, select the optimal threshold for specified accuracy measure


    Parameters
    ----------
    x : 2D ndarray of floats
        DESCRIPTION. Matrix with data of class 0. Each row contains one
        mobservation, each column contains one variable
    y : 2D ndarray of floats
        DESCRIPTION. Matrix with data of class 1. Each row contains one
        mobservation, each column contains one variable
    names : defined in function oneDClass
        DESCRIPTION. defined and used in function oneDClass.
    acc : defined in function oneDClass
        DESCRIPTION. defined and used in function oneDClass.

    Returns
    -------
    bestT float
        DESCRIPTION. Optimal threshold value.
    bestErr float
        DESCRIPTION. best Err is minimal error which corresponds to threshold
        bestT. Error is one minus accuracy defined by acc
    direct 1dimensional ndarray of floats
        DESCRIPTION. dir is vector of fisher direction
    '''

    # Calculate means
    uMean = np.mean(y, axis=0)
    nMean = np.mean(x, axis=0)
    # Calculate covariance matrices
    uCov = np.cov(y, rowvar=False)
    nCov = np.cov(x, rowvar=False)
    # Matrix correction for degenerated case
    mat = uCov + nCov
    mat = mat + 0.001 * np.max(np.abs(mat.diagonal())) * np.identity(uMean.shape[0])

    # Calculate Fisher directions
    direct = np.linalg.solve(mat, uMean - nMean)
    # Check the empty direction
    d = np.sqrt(np.sum(np.square(direct)))
    if abs(d) < np.mean(np.abs(uMean)) * 1.e-5:
        return 0, float('inf'), direct

    # Normalise Fisher direction
    direct = direct / d

    # Calculate projection
    projX = np.matmul(x, direct)
    projY = np.matmul(y, direct)
    # Calculate threshold and return result
    (bestT, bestErr) = oneDClass(projX, projY, names, acc)
    return bestT, bestErr, direct


def oneDClass(x, y, name=None, acc = 'BA'):
    '''
    Function oneDClass applied classification with one input attribute by
    searching the best threshold.

    Parameters
    ----------
    x : TYPE list, tuple or 1D numpy.ndarray
        DESCRIPTION. x contains values for Class 1
    y : TYPE list, tuple or 1D numpy.ndarray
        DESCRIPTION. y contains values for Class 2
    name : list of string or None, optional
        DESCRIPTION. If name is not None then it should contain three elements,
        used in titles on graphs:
            name(1) is name of attribute
            name(2) is name of the first class
            name(3) is name of the second class
        If name is None then graphs are not formed.
        The default is None.
   acc : is string or accuracy measure function.
        DESCRIPTION. For string there are several appropriate values:
            'BA' means balanced accuracy: (TP / (TP + FN) + NT / (TN + FP)) / 2
            'accuracy' means (TP + TN) / (TP + FN + TN + FP)
            'f1' means F1 score 2 * TP / (2 * TP + FN + FP)
            'NPV' means Negative predictive value TN / (TN + FN)
            'PPV' means Positive predictive value TP / (TP + FP)
            'TPR', 'recall', 'sens', 'power' means true positive rate (recall,
                 probability of detection, hit rate, power) TP / (TP + FN)
            'TNR', 'spec', 'sel', means true negative rate (specificity,
                 selectivity) TN / (TN + FP)
        Function as argument must have following syntaxis:
            def funcName(TP, FP, TN, FN) :/
            where TP means true positive, FP means false positive, TN means
            true negative, FN means false negative.
            Function must returm one real value

    Raises
    ------
    RuntimeError
        DESCRIPTION. Raised if argument are inappropriate. Text of
        exception described what should be sent to method.

    Returns
    -------
    bestT is optimal threshold
    bestErr is minimal error which corresponds to threshold bestT. Error is
       one minus accuracy defined by parameter acc.
    '''
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals

    # Check of accuracy and convert to function
    if isinstance(acc, str):
        acc = acc.lower() + '   '
        acc = acc[0:3]
        if acc == 'ba ':
            acc = ba
        elif acc == 'acc':
            acc = accuracy
        elif acc == 'f1 ':
            acc = f1
        elif acc == 'npv':
            acc = npv
        elif acc == 'ppv':
            acc = ppv
        elif acc in ('tpr', 'rec', 'sen', 'pow'):
            acc = tpr
        elif acc in ('tnr', 'sel', 'spe'):
            acc = tnr
        else:
            raise RuntimeError('Inacceptable value of requested accuracy. See function oneDClass')

    # Convert x and y to arrays
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    # Check that x and y are 1D
    if x.ndim != 1 or y.ndim !=1:
        raise RuntimeError("Sorry, but x and y must be 1D array like (list, tuple,numpy.ndarray)")

    # Define numbers of cases
    Pos = x.shape[0]
    Neg = y.shape[0]
    tot = Pos + Neg

    # Define set of unique values
    thr = np.unique(np.concatenate((x, y)))
    # Add two boders
    if thr[0] > 0:
        a = thr[0] * 0.9999
    else:
        a = thr[0] * 1.0001
    if thr[-1] < 0:
        b = thr[-1] * 0.9999
    else:
        b = thr[-1] * 1.0001
    thr = np.concatenate(([a], (thr[0:-1] + thr[1:]) / 2, [b]))
    # Create array for errors
    errs = np.zeros((thr.shape[0]))

    # Define meaning of "class 1"
    xLt =  np.mean(x) > np.mean(y)

    # Define variables to search
    bestErr = tot
    bestT = -np.inf
    # Check each threshold
    for k in range(thr.shape[0]):
        t = thr[k]
        nX = sum(x < t)
        nY = sum(y >= t)
        if xLt:
            nX = Pos - nX
            nY = Neg - nY
        err = 1 - acc(nX, Neg - nY, nY, Pos - nX)
        if err < bestErr:
            bestErr = err
            bestT = t
        errs[k] = err

    # Now we are ready to form graph, if requested
    if not name is None:
        # Define min and max to form bines
        mi = min(np.concatenate((x, y)))
        ma = max(np.concatenate((x, y)))
        edges = np.linspace(mi, ma, 21)

        # Draw histograms
        plt.figure()
        # histogram(x, edges, 'Normalization','probability')
        plt.hist(x, bins=edges, density=True, alpha=0.5, label=name[1])
        # hold on
        # histogram(y, edges, 'Normalization','probability')
        plt.hist(y, bins=edges, density=True, alpha=0.5, label=name[2])
        plt.title(name[0])
        plt.xlabel("Value of " + name[0])
        plt.ylabel('Fraction of cases')
        # Draw graph of errors
        sizes = plt.axis()
        plt.plot(thr, errs * sizes[3], 'g', label="Error")
        # Draw the best threshold
        plt.axvline(bestT , color = 'k', label = 'Threshold')
        # plot([bestT, bestT], sizes(3:4), 'k', 'LineWidth', 2)
        plt.legend()

    return (bestT, bestErr)


# Service functions for calculation of classification quality.
# TP means true positive, FP means false positive,
# TN means true negative, FN means false negative

def ba(TP, FP, TN, FN):
    '''
    Calculated balanced accuracy

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. Balanced accuracy

    '''
    return (TP / (TP + FN) +TN / (TN + FP)) / 2

def accuracy(TP, FP, TN, FN):
    '''
    Calculated accuracy

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. Accuracy

    '''
    return (TP + TN) / (TP + FN + TN + FP)

def f1(TP, FP, _TN, FN):
    '''
    Calculated f1 score

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. f1 score

    '''
    return 2 * TP / (2 * TP + FN + FP)

def npv(_TP, _FP, TN, FN):
    '''
    Calculated Negative predictive value

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. Negative predictive value

    '''
    return TN / (TN + FN)

def ppv(TP, FP, _TN, _FN):
    '''
    Calculated Positive predictive value

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. Positive predictive value

    '''
    return TP / (TP + FP)

def tpr(TP, _FP, _TN, FN):
    '''
    Calculated True positive rate (recall, probability of detection, hit rate,
                                   power)

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. True positive rate

    '''
    return TP / (TP + FN)

def tnr(_TP, FP, TN, _FN):
    '''
    Calculated True negative rate (specificity, selectivity)

    Parameters
    ----------
    TP : float
        DESCRIPTION. True Positive
    FP : float
        DESCRIPTION. FalsePpositive
    TN : float
        DESCRIPTION. True Negative
    FN : float
        DESCRIPTION. False Negative

    Returns
    -------
    float
        DESCRIPTION. True negative rate

    '''
    return TN / (TN + FP)
