import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from prior import Prior  #


class ParametricPrior(Prior):
    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            kernel=None,
            **gp_kwargs
    ):
        super(ParametricPrior, self).__init__(
            X_train=X_train,
            y_train=y_train,
        )
        
        unsupported_keys = ['num_gradient_updates']
        for key in unsupported_keys:
            if key in gp_kwargs:
                del gp_kwargs[key]
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        
       
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        
        
        self.gp = GaussianProcessRegressor(kernel=kernel, **gp_kwargs)
        
        
        self.gp.fit(X_train, y_train)
    
    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        
        X_test = self.scaler.transform(X)
        
        
        mu, sigma = self.gp.predict(X_test, return_std=True)
     
        return mu, sigma

