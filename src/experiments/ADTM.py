import os
import logging
from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
from blackbox import BlackboxOffline
from blackbox.load_utils import evaluation_split_from_task
from optimizer.benchmark import benchmark
from optimizer.gaussian_process_functional_prior import G3P
from optimizer.gaussian_process_functional_prior-improve import IG3P
from optimizer.gaussian_process import GP
from optimizer.random_search import RS
from optimizer.thompson_sampling_functional_prior import TS

def calculate_normalized_dtm(y_opt, y_min, y_max):
    return (y_opt - y_min) / (y_max - y_min)

def evaluate(
        task: str,
        optimizer: str,
        prior: str,
        num_seeds: int,
        num_evaluations: int,
        output_folder: Path,
):
    optimizers = {
        "GP": partial(GP, normalization="standard"),
        "GCP": partial(GP, normalization="gaussian"),
        "RS": RS,
        "IGCP + prior": partial(IG3P, normalization="gaussian"),
        "GCP + prior": partial(G3P, normalization="gaussian", prior=prior),
        "TS": partial(TS, normalization="standard", prior=prior),
        "CTS": partial(TS, normalization="gaussian", prior=prior),
    }

    logging.info(f"Evaluating {optimizer} on {task} with {num_seeds} seeds and {num_evaluations} evaluations.")

    Xys_train, (X_test, y_test) = evaluation_split_from_task(test_task=task)
    candidates = X_test

    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )

    
    y_all = np.concatenate([y for _, y in Xys_train] + [y_test])
    y_min = y_all.min()
    y_max = y_all.max()

    optimizer_factory = partial(
        optimizers[optimizer],
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        evaluations_other_tasks=Xys_train,
    )

    X, y = benchmark(
        optimizer_factory=optimizer_factory,
        blackbox=blackbox,
        candidates=candidates,
        num_seeds=num_seeds,
        num_evaluations=num_evaluations,
        verbose=False,
    )

    y = y.squeeze(axis=-1)

    results = []
    for seed in range(num_seeds):
        y_opt = float('inf')
        for iteration in range(num_evaluations):
            y_opt = min(y_opt, y[seed, iteration])
            dtm = calculate_normalized_dtm(y_opt, y_min, y_max)
            results.append({
                "seed": seed,
                "iteration": iteration,
                "value": y[seed, iteration],
                "blackbox": "XGboost",
                "task": task,
                "optimizer": optimizer,
                "dtm": dtm
            })

    df = pd.DataFrame(results)

    return df

if __name__ == '__main__':
    task = "w6a"
    logging.basicConfig(level=logging.INFO)
    num_seeds = 20
    num_evaluations = 70
    output_folder = Path.cwd() / "experiments" / "ADTM" 
    prior = "pytorch"

    Xys_train, (X_test, y_test) = evaluation_split_from_task(task)
    candidates = X_test

    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )

    optimizers = {
        "IGCP + prior": partial(IG3P, normalization="gaussian"),
        "GCP + prior": partial(G3P, normalization="gaussian"),
        "CTS": partial(TS, normalization="gaussian"),
        "GCP": partial(GP, normalization="gaussian"),
        "RS": RS,
        "TS": TS,
        "GP": GP,
    }

    all_results = []

    for name, Optimizer_cls in optimizers.items():
        logging.info(f"Evaluating {name}")
        optimizer_factory = partial(
            Optimizer_cls,
            input_dim=blackbox.input_dim,
            output_dim=blackbox.output_dim,
            evaluations_other_tasks=Xys_train,
        )
        X, y = benchmark(
            optimizer_factory=optimizer_factory,
            blackbox=blackbox,
            candidates=candidates,
            num_seeds=num_seeds,
            num_evaluations=num_evaluations,
            verbose=False,
        )
        df_results = evaluate(
            task=task,
            optimizer=name,
            prior="pytorch",
            num_seeds=num_seeds,
            num_evaluations=num_evaluations,
            output_folder=output_folder,
        )
        all_results.append(df_results)

    # Merge the results of all optimizers
    final_df = pd.concat(all_results, ignore_index=True)

    # save results
    output_file = output_folder / f"{task}_results.csv"
    final_df.to_csv(output_file, index=False)

    logging.info(f"Results saved to {output_file}")

    # Calculate and print the average ADTM value for each optimizer at the last iteration
    for name in optimizers.keys():
        avg_adtm = final_df[(final_df["optimizer"] == name) & (final_df["iteration"] == 69)]["dtm"].mean()
        print(f"Average ADTM for {name} at last iteration: {avg_adtm}")

