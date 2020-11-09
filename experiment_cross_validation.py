import numpy as np
import argparse
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA

from ope_estimators import OPEestimators


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Off-policy Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='satimage',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    parser.add_argument('--method', '-m', type=str, default=None,
                        choices=['NW_regression', 'knn_regression'])
    args = parser.parse_args(arguments)

    return args


def data_generation(data_name, N):
    X, Y = load_svmlight_file('data/{}'.format(data_name))
    X = X.toarray()
    maxX = X.max(axis=0)
    maxX[maxX == 0] = 1
    X = X / maxX
    Y = np.array(Y, np.int64)

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1
    elif data_name == 'vehicle.scale':
        Y = Y - 1
    elif data_name == 'mnist':
        pca = PCA(n_components=100).fit(X)
        X = pca.transform(X)
    elif data_name == 'letter.scale':
        Y = Y - 1
    elif data_name == 'Sensorless':
        Y = Y - 1
    elif data_name == 'connect-4':
        Y = Y + 1

    classes = np.unique(Y)

    Y_matrix = np.zeros(shape=(N, len(classes)))

    for i in range(N):
        Y_matrix[i, Y[i]] = 1

    return X, Y, Y_matrix, classes


def fit_logreg(X, Y):
    return LogisticRegression(random_state=0, penalty='l2', C=0.1, solver='saga', multi_class='multinomial').fit(X, Y)


def behavior_policy(X, classes, classifier, alpha=0.7):
    N = len(X)
    num_class = len(classes)

    predict = np.array(classifier.predict(X), np.int64)

    pi_predict = np.zeros(shape=(N, num_class))

    for i in range(N):
        pi_predict[i, predict[i]] = 1

    pi_random = np.random.uniform(size=(N, num_class))

    pi_random = pi_random.T
    pi_random /= pi_random.sum(axis=0)
    pi_random = pi_random.T

    pi_behavior = alpha * pi_predict + (1 - alpha) * pi_random

    return pi_behavior


def evaluation_policy(X, A, R, classes, method_list):
    N = len(X)
    num_class = len(classes)

    pi_random = np.random.uniform(size=(N, num_class))

    pi_random = pi_random.T
    pi_random /= pi_random.sum(axis=0)
    pi_random = pi_random.T

    pi_evaluation_list = []

    for clf in method_list:
        clf.fit(X, A, sample_weight=R)
        pi_evaluation = 0.9 * clf.predict_proba(X) + 0.1 * pi_random
        pi_evaluation_list.append(pi_evaluation)

    return pi_evaluation_list


def true_value(Y_matrix, pi_evaluation):
    return np.sum(Y_matrix * pi_evaluation) / len(pi_evaluation)


def cross_validation(X, A, Y, Y_observed_matrix, classes, pi_behavior, method_list, folds=2):
    N_hist = len(X)

    cv_fold = np.arange(folds)
    cv_split0 = np.floor(np.arange(len(X)) * folds / N_hist)
    cv_index = cv_split0[np.random.permutation(N_hist)]

    x_cv = []
    a_cv = []
    y_mat_cv = []
    y_cv = []
    pi_bhv_cv = []

    score_cv_ipw = np.zeros(len(method_list))
    score_cv_dm = np.zeros(len(method_list))
    score_cv_dm_ML = np.zeros(len(method_list))
    score_cv_dml = np.zeros(len(method_list))

    for k in cv_fold:
        x_cv.append(X[cv_index == k])
        a_cv.append(A[cv_index == k])
        y_cv.append(Y[cv_index == k])
        y_mat_cv.append(Y_observed_matrix[cv_index == k])

        pi_bhv_cv.append(pi_behavior[cv_index == k])

    for k in range(folds):
        count = 0
        for j in range(folds):
            if j == k:
                x_te = x_cv[j]
                a_te = a_cv[j]
                y_te = y_cv[j]
                y_mat_te = y_mat_cv[j]
                p_bhv_te = pi_bhv_cv[j]

            if j != k:
                if count == 0:
                    x_tr = x_cv[j]
                    y_tr = y_cv[j]
                    a_tr = a_cv[j]
                    y_mat_tr = y_mat_cv[j]
                    pi_bhv_tr = pi_bhv_cv[j]
                    count += 1
                else:
                    x_tr = np.append(x_tr, x_cv[j], axis=0)
                    y_tr = np.append(y_tr, y_cv[j], axis=0)
                    a_tr = np.append(a_tr, a_cv[j], axis=0)
                    y_mat_tr = np.append(y_mat_tr, y_mat_cv[j], axis=0)
                    pi_bhv_tr = np.append(pi_bhv_tr, pi_bhv_cv[j], axis=0)

        pi_evaluation_list = evaluation_policy(x_tr,
                                               np.argmax(a_tr, axis=1),
                                               np.sum(y_mat_tr, axis=1),
                                               classes, method_list)

        for pi_evaluation_idx, pi_evaluation in enumerate(pi_evaluation_list):
            OPE = OPEestimators(classes, p_bhv_te, pi_evaluation)

            theta_ipw = OPE.fit(x_te, a_te, y_mat_te, est_type='ipw')
            score_cv_ipw[pi_evaluation_idx] += theta_ipw

            theta_dm = OPE.fit(x_te, a_te, y_mat_te, est_type='dm')
            score_cv_dm[pi_evaluation_idx] += theta_dm

            theta_dm_ML = OPE.fit(x_te, a_te, y_mat_te, est_type='dm_ML')
            score_cv_dm_ML[pi_evaluation_idx] += theta_dm_ML

            theta_dml = OPE.fit(x_te, a_te, y_mat_te, est_type='aipw')
            score_cv_dml[pi_evaluation_idx] += theta_dml

    pi_evaluation_list = evaluation_policy(X,
                                           np.argmax(A, axis=1),
                                           np.sum(Y_observed_matrix, axis=1),
                                           classes, method_list)

    (pi_evaluation_idx_chosen_ipw,) = np.unravel_index(
        np.argmax(score_cv_ipw), score_cv_ipw.shape)
    (pi_evaluation_idx_chosen_dm,) = np.unravel_index(
        np.argmax(score_cv_dm), score_cv_dm.shape)
    (pi_evaluation_idx_chosen_dm_ML,) = np.unravel_index(
        np.argmax(score_cv_dm_ML), score_cv_dm_ML.shape)
    (pi_evaluation_idx_chosen_dml,) = np.unravel_index(
        np.argmax(score_cv_dml), score_cv_dml.shape)

    evaluation_chosen_list = [pi_evaluation_idx_chosen_ipw, pi_evaluation_idx_chosen_dm,
                              pi_evaluation_idx_chosen_dm_ML, pi_evaluation_idx_chosen_dml]

    score_matrix = [score_cv_ipw, score_cv_dm, score_cv_dm_ML, score_cv_dml]
    score_matrix_min_method = np.min(score_matrix, axis=0)
    score_matrix_min_method_max_evaluation = np.argmax(score_matrix_min_method)

    return pi_evaluation_list, evaluation_chosen_list, score_matrix_min_method_max_evaluation


def sample_by_behavior(Y_matrix, pi_behavior, sample_size, classes):
    Y_observed_matrix = np.zeros(shape=(sample_size, len(classes)))
    A_observed_matrix = np.zeros(shape=(sample_size, len(classes)))

    for i in range(sample_size):
        a = np.random.choice(classes, p=pi_behavior[i])
        Y_observed_matrix[i, a] = Y_matrix[i, a]
        A_observed_matrix[i, a] = 1

    return Y_observed_matrix, A_observed_matrix


def main(arguments):
    args = process_args(arguments)

    logit_model = LogisticRegression(random_state=0)
    logit_model = GridSearchCV(logit_model, {'solver': ['saga'], 'multi_class': [
                               'multinomial'], 'C': [0.01, 0.1, 1]}, cv=2)

    svc_model = svm.SVC(random_state=0, probability=True)
    svc_model_linear = GridSearchCV(
        svc_model, {'kernel': ['linear'], 'C': [0.01, 0.1, 1]}, cv=2)
    svc_model_poly = GridSearchCV(
        svc_model, {'kernel': ['poly'], 'C': [0.01, 0.1, 1]}, cv=2)
    svc_model_rbf = GridSearchCV(svc_model, {'kernel': ['rbf'], 'C': [
                                 0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1]}, cv=2)

    random_forest = RandomForestClassifier(random_state=0)
    random_forest = GridSearchCV(random_forest, {'max_depth': [
                                 5, 10, 15, 20], 'n_estimators': [10, 50, 100]}, cv=2)

    method_list = [logit_model, svc_model_linear,
                   svc_model_poly, svc_model_rbf, random_forest]

    data_name = args.dataset
    num_trials = args.num_trials
    sample_size = args.sample_size

    alphas = [0.7, 0.4, 0.0]
    behavior_train_size = 1000

    res_ipw_list = np.zeros((num_trials, len(alphas)))
    res_dm_list = np.zeros((num_trials, len(alphas)))
    res_dm_ml_list = np.zeros((num_trials, len(alphas)))
    res_aipw_list = np.zeros((num_trials, len(alphas)))

    res_minimax_list = np.zeros((num_trials, len(alphas)))

    if not os.path.isdir('exp_results'):
        os.makedirs('exp_results')

    for trial in range(num_trials):
        # sample N data {(X, Y)}
        X, Y, Y_true_matrix, classes = data_generation(
            data_name, behavior_train_size + sample_size)
        X_tr, Y_tr = X[:behavior_train_size], Y[:behavior_train_size]
        X_ev, Y_ev, Y_true_matrix_ev = X[behavior_train_size:
                                         ], Y[behavior_train_size:], Y_true_matrix[behavior_train_size:]
        classifier = fit_logreg(X_tr, Y_tr)

        for idx_alpha in range(len(alphas)):
            # construct behavior policy via sample
            alpha = alphas[idx_alpha]
            pi_behavior = behavior_policy(
                X_ev, classes, classifier, alpha=alpha)

            # generate logging data
            Y_observed_matrix, A_observed_matrix = sample_by_behavior(
                Y_true_matrix_ev, pi_behavior, sample_size, classes)
            # estimate the best policy
            pi_evaluation_list, evaluation_chosen_list, evaluation_minimax_chosen = cross_validation(
                X_ev, A_observed_matrix, Y_ev, Y_observed_matrix, classes, pi_behavior, method_list, folds=2)

            true_taus = []

            for pi_evaluation in pi_evaluation_list:
                true_taus.append(true_value(Y_true_matrix_ev, pi_evaluation))

            max_tau = np.max(true_taus)

            res_ipw_list[trial, idx_alpha] = max_tau - \
                true_taus[evaluation_chosen_list[0]]
            res_dm_list[trial, idx_alpha] = max_tau - \
                true_taus[evaluation_chosen_list[1]]
            res_dm_ml_list[trial, idx_alpha] = max_tau - \
                true_taus[evaluation_chosen_list[2]]
            res_aipw_list[trial, idx_alpha] = max_tau - \
                true_taus[evaluation_chosen_list[3]]

            res_minimax_list[trial, idx_alpha] = max_tau - \
                true_taus[evaluation_minimax_chosen]

    np.savetxt("exp_results/res_ipw_data_%s_samplesize_%d.csv" %
               (data_name, sample_size), res_ipw_list, delimiter=",")
    np.savetxt("exp_results/res_dm_sn_data_%s_samplesize_%d.csv" %
               (data_name, sample_size), res_dm_list, delimiter=",")
    np.savetxt("exp_results/res_dm_ML_data_%s_samplesize_%d.csv" %
               (data_name, sample_size), res_dm_ml_list, delimiter=",")
    np.savetxt("exp_results/res_aipw_data_%s_samplesize_%d.csv" %
               (data_name, sample_size), res_aipw_list, delimiter=",")
    np.savetxt("exp_results/res_minimax_data_%s_samplesize_%d.csv" %
               (data_name, sample_size), res_minimax_list, delimiter=",")


if __name__ == '__main__':
    main(sys.argv[1:])
