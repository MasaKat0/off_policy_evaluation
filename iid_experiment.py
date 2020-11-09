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

alphas = [0.7, 0.4, 0.1]


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Off-policy Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='satimage',
                        help='Name of dataset')
    parser.add_argument('--history_sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=100,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
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

    if data_name == 'satimage':
        Y = Y - 1
    elif data_name == 'vehicle':
        Y = Y - 1
    elif data_name == 'mnist':
        pca = PCA(n_components=100).fit(X)
        X = pca.transform(X)
    elif data_name == 'letter':
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


def create_policy(X, classes, classifier, alpha=0.7):
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

    policy = alpha * pi_predict + (1 - alpha) * pi_random

    return policy


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


def sample_by_behavior(Y_matrix, pi_behavior, classes):
    sample_size = len(Y_matrix)
    Y_observed_matrix = np.zeros(shape=(sample_size, len(classes)))
    A_observed_matrix = np.zeros(shape=(sample_size, len(classes)))

    for i in range(sample_size):
        a = np.random.choice(classes, p=pi_behavior[i])
        Y_observed_matrix[i, a] = Y_matrix[i, a]
        A_observed_matrix[i, a] = 1

    return Y_observed_matrix, A_observed_matrix


def main(arguments):
    args = process_args(arguments)

    if not os.path.isdir('exp_results'):
        os.makedirs('exp_results')

    evaluation_policy_size = 1000

    data_name = args.dataset
    num_trials = args.num_trials
    history_sample_size = args.history_sample_size

    true_val_list = np.zeros((num_trials))

    res_ipw_list = np.zeros((num_trials, len(alphas)))
    res_dm_list = np.zeros((num_trials, len(alphas)))
    res_aipw_list = np.zeros((num_trials, len(alphas)))
    res_aipw_ddm_list = np.zeros((num_trials, len(alphas)))
    res_a2ipw_list = np.zeros((num_trials, len(alphas)))

    res_est_ipw_list = np.zeros((num_trials, len(alphas)))
    res_dr_list = np.zeros((num_trials, len(alphas)))
    res_dr_ddm_list = np.zeros((num_trials, len(alphas)))
    res_adr_list = np.zeros((num_trials, len(alphas)))

    for tr in range(num_trials):
        X, Y_true, Y_true_matrix, classes = data_generation(
            data_name, evaluation_policy_size + history_sample_size)
        X_pre = X[:evaluation_policy_size]
        Y_pre_true = Y_true[:evaluation_policy_size]
        _ = Y_true_matrix[:evaluation_policy_size]

        X_hist = X[evaluation_policy_size:]
        _ = Y_true[evaluation_policy_size:]
        Y_hist_true_matrix = Y_true_matrix[evaluation_policy_size:]

        classifier = fit_logreg(X_pre, Y_pre_true)

        pi_evaluation = create_policy(
            X_hist, classes, classifier, alpha=0.9)

        true_val_list[tr] = true_value(
            Y_hist_true_matrix, pi_evaluation)

        np.savetxt("exp_results/true_data_%s_samplesize_%d.csv" %
                   (data_name, history_sample_size), true_val_list, delimiter=",")

        for idx_alpha in range(len(alphas)):
            # construct behavior policy via sample
            alpha = alphas[idx_alpha]
            pi_behavior = create_policy(
                X_hist, classes, classifier, alpha=alpha)

            # generate logging data
            Y_hist, A_hist = sample_by_behavior(
                Y_hist_true_matrix, pi_behavior, classes)

            OPE = OPEestimators(
                classes, pi_evaluation=pi_evaluation,
                pi_behavior=pi_behavior, variance=False)

            res_ipw_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='ipw')
            res_dm_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='dm')
            res_aipw_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='aipw')
            res_aipw_ddm_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='aipw_ddm')
            res_a2ipw_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='a2ipw')

            OPE = OPEestimators(
                classes, pi_evaluation=pi_evaluation,
                pi_behavior=None, variance=False)

            res_est_ipw_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='ipw')
            res_dr_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='dr')
            res_dr_ddm_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='dr_ddm')
            res_adr_list[tr, idx_alpha] = OPE.fit(
                X_hist, A_hist, Y_hist, est_type='adr')

            np.savetxt("exp_results/res_ipw_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_ipw_list, delimiter=",")
            np.savetxt("exp_results/res_dm_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_dm_list, delimiter=",")
            np.savetxt("exp_results/res_aipw_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_aipw_list, delimiter=",")
            np.savetxt("exp_results/res_aipw_ddm_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_aipw_ddm_list, delimiter=",")
            np.savetxt("exp_results/res_a2ipw_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_a2ipw_list, delimiter=",")

            np.savetxt("exp_results/res_est_ipw_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_est_ipw_list, delimiter=",")
            np.savetxt("exp_results/res_dr_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_dr_list, delimiter=",")
            np.savetxt("exp_results/res_dr_ddm_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_dr_ddm_list, delimiter=",")
            np.savetxt("exp_results/res_adr_data_%s_samplesize_%d.csv" %
                       (data_name, history_sample_size),
                       res_adr_list, delimiter=",")


if __name__ == '__main__':
    main(sys.argv[1:])
