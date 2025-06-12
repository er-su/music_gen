from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import numpy as np
from tqdm.auto import tqdm

class IncrementalRBFSVC(BaseEstimator, ClassifierMixin):
    """
    Approximate RBF-kernel SVM using random Fourier features and incremental learning.

    This estimator scales input features, transforms them via random Fourier features
    for an RBF approximation, and fits an SGDClassifier in chunks for memory efficiency.

    Parameters
    ----------
    gamma : float or {'auto'}, default='auto'
        Kernel coefficient for the RBF kernel. If 'auto', uses 1 / n_features.
    n_components : int, default=500
        Number of Monte Carlo samples per original feature for the random Fourier features.
    C : float, default=1.0
        Inverse regularization strength; must be positive.
    loss : {'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'}, default='hinge'
        Loss function for the SGDClassifier.
    max_iter : int, default=1
        Maximum number of iterations over each chunk during partial_fit.
    tol : float or None, default=None
        Tolerance for stopping criteria of SGDClassifier.
    random_state : int or RandomState instance, default=None
        Random seed for reproducibility of RBFSampler and SGDClassifier.
    chunk_size : int, default=10000
        Number of samples to process per chunk when calling fit.
    verbose : bool, default=False
        If True, displays a progress bar during fit using tqdm.
    score : string, default='accuracy'
        Determines the type of accuracy score returned by self-scoring.

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted scaler for input features.
    rbf_sampler_ : RBFSampler
        Fitted random Fourier feature transformer.
    clf_ : SGDClassifier
        Fitted SGD classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features,), dtype=object
        Names of features seen during fit, if X has columns.
    """
    def __init__(self, gamma='auto', n_components=500, C=1.0,
                 loss='hinge', max_iter=1, tol=None, random_state=None,
                 chunk_size=10000, verbose=False, scoring = 'accuracy'):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.scoring = scoring

        # Attributes to be set on first fit/partial_fit
        self.scaler_ = None
        self.rbf_sampler_ = None
        self.clf_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the model on a batch of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        classes : array-like of shape (n_classes,), optional
            List of all possible class labels. Must be provided on first call.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If `classes` is None on the first call.
        """
        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]
            if hasattr(X, "columns"):
                self.feature_names_in_ = np.array(X.columns, dtype=object)
        X = np.asarray(X)
        y = np.asarray(y)

        if self.classes_ is None:
            if classes is None:
                raise ValueError("classes must be provided on first partial_fit")
            self.classes_ = np.asarray(classes)

        if self.scaler_ is None:
            self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        if self.rbf_sampler_ is None:
            gamma = 1.0 / X.shape[1] if self.gamma == 'auto' else self.gamma
            self.rbf_sampler_ = RBFSampler(
                gamma=gamma,
                n_components=self.n_components,
                random_state=self.random_state
            )
            Z = self.rbf_sampler_.fit_transform(Xs)
        else:
            Z = self.rbf_sampler_.transform(Xs)
        alpha = 1.0 / self.C
        if self.clf_ is None:
            self.clf_ = SGDClassifier(
                loss=self.loss, penalty='l2', alpha=alpha,
                max_iter=self.max_iter, tol=self.tol,
                random_state=self.random_state
            )
            self.clf_.partial_fit(Z, y, classes=self.classes_)
        else:
            self.clf_.partial_fit(Z, y)

        return self

    def fit(self, X, y):
        """
        Fit the model by splitting data into chunks and calling partial_fit sequentially.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_samples = X.shape[0]
        first = True

        iterator = range(0, n_samples, self.chunk_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Fitting chunks", unit="chunk")

        for start in iterator:
            end = start + self.chunk_size
            X_chunk = X[start:end]
            y_chunk = y[start:end]

            if first:
                self.partial_fit(X_chunk, y_chunk, classes=classes)
                first = False
            else:
                self.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        Xs = self.scaler_.transform(X)
        Z = self.rbf_sampler_.transform(Xs)
        return self.clf_.predict(Z)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probability estimates.

        Raises
        ------
        AttributeError
            If the underlying SGDClassifier does not support probability estimates.
        """
        if not hasattr(self.clf_, "predict_proba"):
            raise AttributeError(
                f"predict_proba is not available for loss='{self.loss}'. "
                "Use loss='log' or 'modified_huber', or wrap in a calibrator."
            )
        X = np.asarray(X)
        Xs = self.scaler_.transform(X)
        Z = self.rbf_sampler_.transform(Xs)
        return self.clf_.predict_proba(Z)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import jaccard_score, accuracy_score
        if self.scoring == "jaccard":
            y_pred = jaccard_score(y, self.predict(X), average="macro")
        else:
            y_pred = accuracy_score(y, self.predict(X))
        return y_pred
