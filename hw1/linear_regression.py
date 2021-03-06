import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from heapq import nlargest


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')
        y_pred = X@self.weights_  #no need to add bias, you are ok
        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N, n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """

        N = len(X)
        n = X.shape[1]

        I = np.identity(n)
        I[0, 0] = 0
        X, y = check_X_y(X, y)

        left  = np.linalg.inv((1 / N) * X.transpose() @ X + self.reg_lambda * I)
        right = (1/N)*X.transpose()@y
        self.weights_ = left@right
        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        ones = np.ones((len(X), 1))
        return np.append(ones, X, axis=1)

class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        #check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').
        X_transformed = np.delete(X, 0, axis=1) #TODO: make sure the comment is wrong and those are the right indices
        poly = sklearn.preprocessing.PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=True)
        X_transformed = poly.fit_transform(X_transformed)
        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    def get_corr(x):
        x_expect = np.mean(x)
        x_normalized = (x - x_expect).to_numpy()
        nominator = x_normalized.transpose() @ y_normalized
        denominator2 = np.sqrt(x_normalized.transpose() @ x_normalized)
        corr = nominator / (denominator1 * denominator2)
        return corr

    y = df[target_feature]
    y_expect = np.mean(y)
    y_normalized = (y - y_expect).to_numpy()
    denominator1 = np.sqrt(y_normalized.transpose() @ y_normalized)

    correlations = [(name, get_corr(x)) for name, x in df.iteritems() if name != target_feature]

    best_five = nlargest(n=5, iterable=correlations, key=lambda r: abs(r[1]))

    top_n_features = [name for name, corr in best_five]
    top_n_corr = [corr for name, corr in best_five]
    return top_n_features, top_n_corr


def evaluate_accuracy(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean squared error (MSE) and coefficient of determination (R-squared).
    :param y: Target values.
    :param y_pred: Predicted values.
    :return: A tuple containing the MSE and R-squared values.
    """
    mse = np.mean((y - y_pred) ** 2)
    rsq = 1 - mse / np.var(y)
    return mse.item(), rsq.item()


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """
    def k_foldCV(d, lam):
        mod = sklearn.pipeline.make_pipeline(
            BiasTrickTransformer(),
            BostonFeaturesTransformer(degree=d),
            LinearRegressor(reg_lambda=lam)
        )
        loss_for_params = []
        kf = KFold(n_splits=k_folds)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mod.fit(X_train, y_train)
            y_pred = mod.predict(X_test)
            loss_for_params.append(evaluate_accuracy(y_test, y_pred)[0])

        return np.mean(loss_for_params)


    # TODO: Do K-fold cross validation to find the best hyperparameters
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters it has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    losses = []
    for d in degree_range:
        for lam in lambda_range:
            loss = k_foldCV(d, lam)
            losses.append((loss, d, lam))

    params = model.get_params()
    best_paramameters = min(losses, key=lambda r: r[0])
    best_degree = best_paramameters[1]
    best_lambda = best_paramameters[2]
    best_params = {'bostonfeaturestransformer__degree': best_degree, 'linearregressor__reg_lambda': best_lambda}
    return best_params
