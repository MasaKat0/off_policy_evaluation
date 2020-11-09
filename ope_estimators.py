import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from kernel_regression import KernelLogit
import warnings


KernelRidge_hyp_param = {'alpha': [0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1]}
KernelLogit_sigma_list = np.array([0.01, 0.1, 1])
KernelLogit_lda_list = np.array([0.01, 0.1, 1])


def kernel_ridge_estimator(X, Y, Z, cv=2):
    model = KernelRidge(kernel='rbf')
    model = GridSearchCV(
        model, {'alpha': [0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1]}, cv=cv)
    model.fit(X, Y)
    return model.predict(Z)


def kernel_logit_estimator(X, Y, Z, cv=2):
    model, KX, KZ = KernelLogit(X, Y, Z, folds=cv, num_basis=100,
                                sigma_list=KernelLogit_sigma_list,
                                lda_list=KernelLogit_lda_list, algorithm='Ridge')
    model.fit(KX, Y)
    return model.predict_proba(KZ)


class OPEestimators():
    def __init__(self, classes, pi_evaluation, pi_behavior=None, variance=False):
        self.classes = classes
        self.pi_behavior = pi_behavior
        self.pi_evaluation = pi_evaluation
        self.variance = variance

    def fit(self, X, A, Y_matrix, est_type,
            outcome_estimator=kernel_ridge_estimator,
            policy_estimator=kernel_logit_estimator,
            warning_samples=10):
        self.X = X
        self.N_hst, self.dim = X.shape
        self.A = A
        self.Y = Y_matrix
        self.warning_samples = warning_samples

        self.outcome_estimator = kernel_ridge_estimator
        self.policy_estimator = kernel_logit_estimator

        warnings.simplefilter('ignore')

        if est_type == 'ipw':
            theta, var = self.ipw()
        if est_type == 'dm':
            theta, var = self.dm()
        if est_type == 'aipw_ddm':
            theta, var = self.aipw_ddm()
        if est_type == 'aipw':
            theta, var = self.aipw()
        if est_type == 'a2ipw':
            theta, var = self.a2ipw()
        if est_type == 'adr':
            theta, var = self.adr()
        if est_type == 'dr_ddm':
            theta, var = self.dr_ddm()
        if est_type == 'dr':
            theta, var = self.dr()

        if self.variance:
            return theta, var
        else:
            return theta

    def aipw_ddm(self, folds=2):
        theta_list = []

        cv_fold = np.arange(folds)
        cv_split0 = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_index = cv_split0[np.random.permutation(self.N_hst)]

        x_cv = []
        a_cv = []
        y_cv = []
        pi_bhv_cv = []
        pi_evl_cv = []

        for k in cv_fold:
            x_cv.append(self.X[cv_index == k])
            a_cv.append(self.A[cv_index == k])
            y_cv.append(self.Y[cv_index == k])

            pi_bhv_cv.append(self.pi_behavior[cv_index == k])
            pi_evl_cv.append(self.pi_evaluation[cv_index == k])

        for k in range(folds):
            count = 0
            for j in range(folds):
                if j == k:
                    x_te = x_cv[j]
                    a_te = a_cv[j]
                    y_te = y_cv[j]
                    pi_bhv_te = pi_bhv_cv[j]
                    pi_evl_te = pi_evl_cv[j]

                if j != k:
                    if count == 0:
                        x_tr = x_cv[j]
                        a_tr = a_cv[j]
                        y_tr = y_cv[j]
                        pi_bhv_tr = pi_bhv_cv[j]
                        pi_evl_tr = pi_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        pi_bhv_tr = np.append(pi_bhv_tr, pi_bhv_cv[j], axis=0)
                        pi_evl_tr = np.append(pi_evl_tr, pi_evl_cv[j], axis=0)

            densratio_matrix = pi_evl_te/pi_bhv_te

            f_matrix = np.zeros(shape=(len(x_te), len(self.classes)))

            for c in self.classes:
                f_matrix[:, c] = self.outcome_estimator(
                    x_tr[a_tr[:, c] == 1], y_tr[:, c][a_tr[:, c] == 1], x_te)

            # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
            weight = len(a_te)
            theta = np.sum(a_te*(y_te-f_matrix)*densratio_matrix /
                           weight) + np.sum(f_matrix*pi_evl_te/weight)

            theta_list.append(theta)

        theta = np.mean(theta_list)

        densratio_matrix = self.pi_evaluation/self.pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            for t in range(self.N_hst):
                if np.sum(self.A[:t, c] == 1) > self.warning_samples:
                    f_matrix[t, c] = self.outcome_estimator(
                        self.X[:t][self.A[:t, c] == 1],
                        self.Y[:t][:t, c][self.A[:t, c] == 1],
                        [self.X[t]])
                else:
                    f_matrix[t, c] = 0

        # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
        score = np.sum(self.A*(self.Y-f_matrix)*densratio_matrix,
                       axis=1) + np.sum(f_matrix*self.pi_evaluation, axis=1)

        var = np.mean((score - theta)**2)

        return theta, var

    def dr_ddm(self, folds=2):
        theta_list = []

        cv_fold = np.arange(folds)
        cv_split0 = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_index = cv_split0[np.random.permutation(self.N_hst)]

        x_cv = []
        a_cv = []
        y_cv = []
        pi_evl_cv = []

        for k in cv_fold:
            x_cv.append(self.X[cv_index == k])
            a_cv.append(self.A[cv_index == k])
            y_cv.append(self.Y[cv_index == k])

            pi_evl_cv.append(self.pi_evaluation[cv_index == k])

        for k in range(folds):
            count = 0
            for j in range(folds):
                if j == k:
                    x_te = x_cv[j]
                    a_te = a_cv[j]
                    y_te = y_cv[j]
                    pi_evl_te = pi_evl_cv[j]

                if j != k:
                    if count == 0:
                        x_tr = x_cv[j]
                        a_tr = a_cv[j]
                        y_tr = y_cv[j]
                        pi_evl_tr = pi_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        pi_evl_tr = np.append(pi_evl_tr, pi_evl_cv[j], axis=0)

            a_temp = np.where(a_tr == 1)[1]
            pi_bhv_te = kernel_logit_estimator(
                x_tr, a_temp, x_te)

            densratio_matrix = pi_evl_te/pi_bhv_te

            f_matrix = np.zeros(shape=(len(x_te), len(self.classes)))

            for c in self.classes:
                f_matrix[:, c] = self.outcome_estimator(
                    x_tr[a_tr[:, c] == 1], y_tr[:, c][a_tr[:, c] == 1], x_te)

            # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
            weight = len(a_te)
            theta = np.sum(a_te*(y_te-f_matrix)*densratio_matrix /
                           weight) + np.sum(f_matrix*pi_evl_te/weight)

            theta_list.append(theta)

        theta = np.mean(theta_list)

        a_temp = np.where(self.A == 1)[1]
        pi_behavior = kernel_logit_estimator(self.X, a_temp, self.X)

        densratio_matrix = self.pi_evaluation/pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            for t in range(self.N_hst):
                if np.sum(self.A[:t, c] == 1) > self.warning_samples:
                    f_matrix[t, c] = self.outcome_estimator(
                        self.X[:t][self.A[:t, c] == 1],
                        self.Y[:t][:t, c][self.A[:t, c] == 1],
                        [self.X[t]])
                else:
                    f_matrix[t, c] = 0

        # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
        score = np.sum(self.A*(self.Y-f_matrix)*densratio_matrix,
                       axis=1) + np.sum(f_matrix*self.pi_evaluation, axis=1)

        var = np.mean((score - theta)**2)

        return theta, var

    def a2ipw(self):
        densratio_matrix = self.pi_evaluation/self.pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            for t in range(self.N_hst):
                if np.sum(self.A[:t, c] == 1) > self.warning_samples:
                    f_matrix[t, c] = self.outcome_estimator(
                        self.X[:t][self.A[:t, c] == 1],
                        self.Y[:t][:t, c][self.A[:t, c] == 1],
                        [self.X[t]])
                else:
                    f_matrix[t, c] = 0

        # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
        score = np.sum(self.A*(self.Y-f_matrix)*densratio_matrix,
                       axis=1) + np.sum(f_matrix*self.pi_evaluation, axis=1)

        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var

    def adr(self):

        pi_behavior = np.copy(self.pi_evaluation)
        pi_behavior[:] = 0.5

        for t in range(1, self.N_hst):
            if all(np.sum(self.A[:t, :] == 1, axis=0) > self.warning_samples):
                a_temp = np.where(self.A[:t] == 1)[1]
                pi_behavior[t, :] = kernel_logit_estimator(
                    self.X[:t], a_temp, np.array([self.X[t]]))

            else:
                pi_behavior[t, :] = 0.5

        densratio_matrix = self.pi_evaluation/pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            for t in range(self.N_hst):
                if np.sum(self.A[:t, c] == 1) > self.warning_samples:
                    f_matrix[t, c] = self.outcome_estimator(
                        self.X[:t][self.A[:t, c] == 1],
                        self.Y[:t][:t, c][self.A[:t, c] == 1],
                        [self.X[t]])
                else:
                    f_matrix[t, c] = 0

        # weight = np.ones(shape=a_te.shape)*np.sum(a_te/pi_bhv_te, axis=0)
        score = np.sum(self.A*(self.Y-f_matrix)*densratio_matrix,
                       axis=1) + np.sum(f_matrix*self.pi_evaluation, axis=1)

        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var

    def ipw(self):
        if self.pi_behavior is None:
            a_temp = np.where(self.A == 1)[1]
            self.pi_behavior = kernel_logit_estimator(self.X, a_temp, self.X)

        densratio = self.pi_evaluation/self.pi_behavior

        # weight = np.ones(shape=self.A.shape)*np.sum(self.A/self.pi_behavior, axis=0)

        score = np.sum(self.A*self.Y*densratio, axis=1)

        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var

    def dr(self):
        a_temp = np.where(self.A == 1)[1]
        pi_behavior = kernel_logit_estimator(self.X, a_temp, self.X)

        densratio = self.pi_evaluation/pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            f_matrix[:, c] = self.outcome_estimator(
                self.X[self.A[:, c] == 1],
                self.Y[:, c][self.A[:, c] == 1],
                self.X)

        # weight = np.ones(shape=self.A.shape)*np.sum(self.A/self.pi_behavior, axis=0)

        score = np.sum(self.A*(self.Y-f_matrix)*densratio, axis=1) + \
            np.sum(f_matrix*self.pi_evaluation, axis=1)

        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var

    def aipw(self):
        densratio = self.pi_evaluation/self.pi_behavior

        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            f_matrix[:, c] = self.outcome_estimator(
                self.X[self.A[:, c] == 1],
                self.Y[:, c][self.A[:, c] == 1],
                self.X)

        # weight = np.ones(shape=self.A.shape)*np.sum(self.A/self.pi_behavior, axis=0)

        score = np.sum(self.A*(self.Y-f_matrix)*densratio, axis=1) + \
            np.sum(f_matrix*self.pi_evaluation, axis=1)

        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var

    def dm(self, method='Ridge'):
        f_matrix = np.zeros(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            f_matrix[:, c] = self.outcome_estimator(
                self.X[self.A[:, c] == 1],
                self.Y[:, c][self.A[:, c] == 1],
                self.X)

        score = np.sum(f_matrix*self.pi_evaluation, axis=1)
        theta = np.mean(score)
        var = np.mean((score - theta)**2)

        return theta, var
