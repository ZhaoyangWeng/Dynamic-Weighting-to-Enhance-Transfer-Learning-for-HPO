import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cho_factor, cho_solve
from constants import num_gradient_updates
from prior import Prior
from sklearn.kernel_approximation import Nystroem


class ParametricPriorSklearn(Prior):
    def __init__(self, X_train: np.array, y_train: np.array, num_gradient_updates: int = num_gradient_updates, n_components: int = 100):
        self.estimator = MLPRegressor(
            activation='relu',
            hidden_layer_sizes=(50, 50, 50),
            learning_rate='adaptive',
            verbose=False,
            max_iter=num_gradient_updates,
            tol=1e-6,
            early_stopping=True,
        )
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.estimator.fit(X_scaled, y_train.ravel())
        
        self.kernel_approx = Nystroem(n_components=n_components)
        X_approx = self.kernel_approx.fit_transform(X_scaled)
        
        self.cholesky_L = self._compute_cholesky(X_approx)
        self.X_train_approx = X_approx  # Store the approximated training data for later use

    def _compute_covariance(self, X1, X2):
        return self.kernel_approx.transform(X1) @ self.kernel_approx.components_.T

    def _compute_cholesky(self, X_train):
        K = self._compute_covariance(X_train, X_train)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
        return L

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_approx = self.kernel_approx.transform(X_scaled)
        mu = self.estimator.predict(X_scaled).reshape((-1, 1))
        sigma = np.ones_like(mu)  # Placeholder, can be refined

        K_s = self._compute_covariance(self.X_train_approx, X_approx)
        L_inv = cho_solve((self.cholesky_L, True), np.eye(self.cholesky_L.shape[0]))
        K_inv = L_inv.T @ L_inv
        cov_pred = K_s.T @ K_inv @ K_s

        return mu, sigma, cov_pred


if __name__ == '__main__':

    num_train_examples = 10000
    num_test_examples = num_train_examples
    dim = 2
    num_gradient_updates = 200
    lr = 1e-2

    def make_random_X_y(num_examples: int, dim: int, noise_std: float):
        X = np.random.rand(num_examples, dim)
        noise = np.random.normal(scale=noise_std, size=(num_examples, 1))
        y = X.sum(axis=-1, keepdims=True) + noise
        return X, y


    # test that parametric prior can recover a simple linear function for the mean
    noise_std = 0.01
    X_train, y_train = make_random_X_y(num_examples=num_train_examples, dim=dim, noise_std=noise_std)
    prior = ParametricPriorSklearn(
        X_train=X_train,
        y_train=y_train,
        num_gradient_updates=num_gradient_updates,
        #num_decays=2,
        # smaller network for UT speed
        #num_layers=2,
        #num_hidden=20,
        #dropout=0.0,
        #lr=lr
    )
    X_test, y_test = make_random_X_y(num_examples=num_test_examples, dim=dim, noise_std=noise_std)
    mu_pred, sigma_pred, cov_pred = prior.predict(X_test)

    mu_l1_error = np.abs(mu_pred - y_test).mean()
    print(mu_l1_error)
    assert mu_l1_error < 0.2
