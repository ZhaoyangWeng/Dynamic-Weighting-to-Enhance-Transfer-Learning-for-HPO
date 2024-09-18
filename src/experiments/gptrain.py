import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path
from blackbox.load_utils import evaluation_split_from_task
import os
from sklearn.preprocessing import StandardScaler
from optimizer.normalization_transforms import from_string

def train_and_save_gp_model(task: str, save_dir: Path):
    """
    Train and save a GP model using data from the specified task.

    Parameters:
    - task: Name of the task to generate data.
    - save_dir: Directory to save the trained GP model.
    """
    logging.info(f"Starting training for task: {task}")

    # Generate example data from the specified task
    Xys_train, _ = evaluation_split_from_task(test_task=task)
    logging.info(f"Generated training data for task: {task}")

    X_source = np.vstack([e[0] for e in Xys_train])
    y_source = np.vstack([e[1] for e in Xys_train])
    logging.info(f"Data shapes - X_source: {X_source.shape}, y_source: {y_source.shape}")

    # Apply standard normalization to X (StandardScaler)
    X_transformer = StandardScaler()
    X_source_transformed = X_transformer.fit_transform(X_source)

    # Apply Gaussian normalization to Y (GaussianTransform)
    y_transformer = from_string("gaussian")(y_source)
    y_source_transformed = y_transformer.transform(y_source)

    logging.info(f"Data standardized with Gaussian normalization for Y and StandardScaler for X")

    # Convert data to torch Tensors
    X_source_tensor = torch.tensor(X_source_transformed, dtype=torch.float32)
    y_source_tensor = torch.tensor(y_source_transformed, dtype=torch.float32)

    logging.info(f"Converted data to torch Tensors")

    # Train GP model
    source_gp = SingleTaskGP(train_X=X_source_tensor, train_Y=y_source_tensor)
    mll = ExactMarginalLogLikelihood(source_gp.likelihood, source_gp)
    logging.info(f"Model and MLL initialized")

    logging.info(f"Initial model parameters: {list(source_gp.parameters())}")

    fit_gpytorch_model(mll)
    logging.info(f"Model training completed")

    logging.info(f"Trained model parameters: {list(source_gp.parameters())}")

    # Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    save_path = save_dir / f'{task}_gp_model_gaussian.pth'
    torch.save({
        'model_state_dict': source_gp.state_dict(),
        'likelihood_state_dict': source_gp.likelihood.state_dict(),
        'X_transformer': X_transformer,
        'y_transformer': y_transformer
    }, save_path)
    logging.info(f"GP model saved to {save_path}")

    # Print the size of the saved model
    model_size = os.path.getsize(save_path) / (1024 * 1024)  # Convert bytes to MB
    logging.info(f"Saved GP model size: {model_size:.2f} MB")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    task = "w6a"   # Example task
    save_dir = Path.home() / 'GPmodels'  # Assuming 'GPmodels' is created in the home directory
    
    # Train and save GP model with Gaussian normalization for Y and StandardScaler for X
    train_and_save_gp_model(task, save_dir)

