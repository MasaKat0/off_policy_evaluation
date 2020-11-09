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
    parser.add_argument('--num_trials', '-n', type=int, default=10,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    parser.add_argument('--policy_type', '-pt', type=str, default=None)
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


class Basic(object):
    def __init__(self, num_arm, T):
        self.num_arm = num_arm
        self.sum_of_reward = np.zeros(num_arm)
        self.num_of_trial = np.zeros(num_arm)
        self.T = T


class UCB(Basic):
    def __init__(self, num_arm, T, dim, sigma2_0=1, sigma2=1, alpha=1):
        super().__init__(num_arm, T)
        self.ucb_score = np.zeros(num_arm)
        self.identity = np.identity(dim)

        self.sigma2_0 = sigma2_0
        self.sigma2 = sigma2

        self.A_inv_array = [(self.sigma2_0/self.sigma2) *
                            self.identity for i in range(num_arm)]

        self.b_array = [np.zeros((dim, 1)) for i in range(num_arm)]

        self.alpha = alpha

    def __call__(self, t, covariate):
        alpha_t = self.alpha*np.sqrt(np.log(t+1))

        for arm in range(self.num_arm):
            theta = self.A_inv_array[arm].dot(self.b_array[arm])
            m0 = covariate.T.dot(theta)
            m1 = alpha_t * \
                np.sqrt(
                    self.sigma2)*np.sqrt(covariate.T.dot(self.A_inv_array[arm]).dot(covariate))
            self.ucb_score[arm] = m0 + m1

        return np.argmax(self.ucb_score)

    def update(self, arm, reward, covariate):
        self.sum_of_reward[arm] += reward
        self.num_of_trial[arm] += 1

        A_inv_temp = self.A_inv_array[arm].copy()
        A_inv_temp0 = A_inv_temp.dot(covariate).dot(
            covariate.T).dot(A_inv_temp)
        A_inv_temp1 = 1+covariate.T.dot(A_inv_temp).dot(covariate)
        self.A_inv_array[arm] -= A_inv_temp0/A_inv_temp1

        self.b_array[arm] += covariate*reward


class TS(Basic):
    def __init__(self, num_arm, T, dim, sigma2_0=1, sigma2=1, alpha=1):
        super().__init__(num_arm, T)
        self.ucb_score = np.zeros(num_arm)
        self.identity = np.identity(dim)

        self.sigma2_0 = sigma2_0
        self.sigma2 = sigma2

        self.A_inv_array = [(self.sigma2_0/self.sigma2) *
                            self.identity for i in range(num_arm)]

        self.b_array = [np.zeros((dim, 1)) for i in range(num_arm)]

        self.alpha = alpha

    def __call__(self, t, covariate):

        for arm in range(self.num_arm):
            mu = self.A_inv_array[arm].dot(self.b_array[arm])

            theta = np.random.multivariate_normal(
                mu.T[0], self.sigma2*self.A_inv_array[arm])

            self.ucb_score[arm] = covariate.T.dot(theta)

        return np.argmax(self.ucb_score)

    def update(self, arm, reward, covariate):
        self.sum_of_reward[arm] += reward
        self.num_of_trial[arm] += 1

        A_inv_temp = self.A_inv_array[arm].copy()
        A_inv_temp0 = A_inv_temp.dot(covariate).dot(
            covariate.T).dot(A_inv_temp)
        A_inv_temp1 = 1+covariate.T.dot(A_inv_temp).dot(covariate)
        self.A_inv_array[arm] -= A_inv_temp0/A_inv_temp1

        self.b_array[arm] += covariate*reward


def create_bandit_policy(X, classes, Y, policy_type='RW', predct_alg='Logit', tau=0.7):
    sample_size, dim = X.shape
    num_actions = len(classes)

    chosen_action_matrix = np.zeros(shape=(sample_size, num_actions))
    observed_outcome_matrix = np.zeros(shape=(sample_size, num_actions))

    if policy_type == 'UCB':
        pi_behavior_array = np.zeros((sample_size, num_actions))
        next_candidate = np.random.uniform(0.01, 0.99, size=(1, num_actions))
        next_candidate = next_candidate/np.sum(next_candidate)
        pi_behavior_array[0] = next_candidate

        ucb = UCB(num_arm=num_actions, T=sample_size,
                  dim=dim, sigma2_0=5, sigma2=5)

        for time in range(sample_size):
            covariate_t = np.array([X[time]]).T

            arm = ucb(time, covariate_t)
            uni_rand = np.random.uniform(size=(num_actions))
            uni_rand = uni_rand/np.sum(uni_rand)
            prob = (1-tau)*uni_rand
            prob[arm] += tau

            pi_behavior_array[time] = prob

            chosen_action = np.random.choice(
                classes, p=pi_behavior_array[time])
            observed_outcome = Y[time, chosen_action]

            chosen_action_matrix[time, chosen_action] = 1
            observed_outcome_matrix[time,
                                    chosen_action] = observed_outcome

            ucb.update(chosen_action,
                       observed_outcome, covariate_t)

    if policy_type == 'TS':
        pi_behavior_array = np.zeros((sample_size, num_actions))
        next_candidate = np.random.uniform(0.01, 0.99, size=(1, num_actions))
        next_candidate = next_candidate/np.sum(next_candidate)
        pi_behavior_array[0] = next_candidate

        ts = TS(num_arm=num_actions, T=sample_size,
                dim=dim, sigma2_0=1, sigma2=1)

        for time in range(sample_size):
            covariate_t = np.array([X[time]]).T

            arm = ts(time, covariate_t)
            uni_rand = np.random.uniform(size=(num_actions))
            uni_rand = uni_rand/np.sum(uni_rand)

            prob = (1-tau)*uni_rand
            prob[arm] += tau

            pi_behavior_array[time] = prob

            chosen_action = np.random.choice(
                classes, p=prob)
            observed_outcome = Y[time, chosen_action]

            chosen_action_matrix[time, chosen_action] = 1
            observed_outcome_matrix[time,
                                    chosen_action] = observed_outcome

            ts.update(chosen_action,
                      observed_outcome, covariate_t)

    return pi_behavior_array, observed_outcome_matrix, chosen_action_matrix


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
    policy_type = args.policy_type

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

        np.savetxt("exp_results/true_data_%s_samplesize_%d_pol_type_%s.csv" %
                   (data_name, history_sample_size, policy_type),
                   true_val_list, delimiter=",")

        for idx_alpha in range(len(alphas)):
            # construct behavior policy via sample
            alpha = alphas[idx_alpha]

            pi_behavior, Y_hist, A_hist = create_bandit_policy(
                X_hist, classes,  Y_hist_true_matrix,
                policy_type=policy_type, tau=alpha)

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

            np.savetxt("exp_results/res_ipw_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_ipw_list, delimiter=",")
            np.savetxt("exp_results/res_dm_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_dm_list, delimiter=",")
            np.savetxt("exp_results/res_aipw_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_aipw_list, delimiter=",")
            np.savetxt("exp_results/res_aipw_ddm_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_aipw_ddm_list, delimiter=",")
            np.savetxt("exp_results/res_a2ipw_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_a2ipw_list, delimiter=",")

            np.savetxt("exp_results/res_est_ipw_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_est_ipw_list, delimiter=",")
            np.savetxt("exp_results/res_dr_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_dr_list, delimiter=",")
            np.savetxt("exp_results/res_dr_ddm_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_dr_ddm_list, delimiter=",")
            np.savetxt("exp_results/res_adr_data_%s_samplesize_%d_pol_type_%s.csv" %
                       (data_name, history_sample_size, policy_type),
                       res_adr_list, delimiter=",")


if __name__ == '__main__':
    main(sys.argv[1:])
