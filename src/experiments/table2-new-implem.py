import os

import pandas as pd
from pathlib import Path

from experiments.load_results import load_results_paper, load_results_reimplem, add_adtm
from experiments.optimizer_names import names
from experiments.table2 import adtm_scores, rank

path = Path(__file__).parent


if __name__ == '__main__':
    df_paper = load_results_paper(do_add_adtm=False)
    df_reimplem = load_results_reimplem()

    df = pd.concat([df_paper, df_reimplem], sort=False)
    print(df.optimizer.unique())
    optimizers_to_plot = [
        "RS",
        names.CTS_prior,
        "CTS (sklearn)",
        "CTS (pytorch)",
        names.GCP_prior,
        "GCP+prior (sklearn)",
        "GCP+prior (pytorch)",
    ]
    df = add_adtm(df)

    scores_per_blackbox, scores_per_task = adtm_scores(df, optimizers_to_plot)

    print(scores_per_blackbox.to_string())
    print(scores_per_blackbox.to_latex(float_format='%.2f', na_rep='-'))
    
    output_dir = path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)


    scores_per_blackbox.to_csv(output_dir / "scores_per_blackbox.csv")
    scores_per_task.to_csv(output_dir / "scores_per_task.csv")


    with open(output_dir / "scores_per_blackbox.tex", "w") as f:
        f.write(scores_per_blackbox.to_latex(float_format='%.2f', na_rep='-'))
    
    with open(output_dir / "scores_per_task.tex", "w") as f:
        f.write(scores_per_task.to_latex(float_format='%.2f', na_rep='-'))
    
    print(f"Results saved to {output_dir}")
