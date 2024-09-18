import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import itertools
import traceback

from evaluate_optimizer_task import evaluate


benchmark_tasks = {
    "nas102": [
        "cifar10",
        "cifar100",
        "ImageNet16-120"
    ],
    "fcnet": [
        "naval",
        "parkinsons",
        "protein",
        "slice"
    ]
}

def run_evaluation(args):
    task, optimizer, prior, num_seeds, num_evaluations, output_folder = args
    try:
        logging.info(f"Evaluating {optimizer} on {task} with {num_seeds} seeds and {num_evaluations} evaluations.")
        result = evaluate(task, optimizer, prior, num_seeds, num_evaluations, output_folder)
        return result
    except Exception as e:
        logging.error(f"Error in evaluating {optimizer} on {task}: {e}")
        traceback.print_exc()
        return None

def main(prior, num_seeds, num_evaluations, output_folder, num_workers):
  
    optimizers = ["GP", "GCP", "RS", "GP+prior", "GCP+prior", "TS", "CTS"]

    
    Path(output_folder).mkdir(parents=True, exist_ok=True)

  
    tasks = []
    for benchmark, task_list in benchmark_tasks.items():
        for task in task_list:
            for optimizer in optimizers:
                tasks.append((task, optimizer, prior, num_seeds, num_evaluations, output_folder))

 
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_evaluation, tasks)

    
    results = [result for result in results if result is not None]

    
    if results:
        
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(Path(output_folder) / "combined_results.csv.zip", index=False)
        logging.info(f"All evaluations are done. Combined results saved to {output_folder}/combined_results.csv.zip")

        
        plot_results(combined_df, output_folder)
    else:
        logging.error("No valid results were obtained from the evaluations.")
    
    #combined_df = pd.concat(results, ignore_index=True)
    #combined_df.to_csv(Path(output_folder) / "combined_results.csv.zip", index=False)
    #logging.info(f"All evaluations are done. Combined results saved to {output_folder}/combined_results.csv.zip")

    
    #plot_results(combined_df, output_folder)


def load_results(file_path):
    df = pd.read_csv(file_path)
    df["ADTM"] = df.groupby(["blackbox", "optimizer", "seed"])["value"].transform(lambda x: x.expanding().mean())
    return df


def plot_optimizers(df, ax, blackbox, optimizers, legend: bool = False):
    df_plot = df.loc[df.optimizer.isin(optimizers), :]

    pivot_df = df_plot.loc[df_plot.blackbox == blackbox, :].groupby(
        ['blackbox', 'optimizer', 'iteration']
    )['ADTM'].mean().reset_index().pivot_table(
        index='iteration', columns='optimizer', values='ADTM'
    ).dropna()

    optimizers = [m for m in optimizers if m in pivot_df]
    style = ["-", "--", "-.", ":", "-", "--", "-."]  
    color = ["b", "g", "r", "c", "m", "y", "k"]  
    pivot_df[optimizers].plot(
        ax=ax,
        title=blackbox,
        color=list(color[:len(optimizers)]),
        style=list(style[:len(optimizers)]),
        markevery=20,
        alpha=0.8,
        lw=2.5,
    )
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylabel('ADTM')
    if not legend:
        ax.get_legend().remove()
    else:
        ax.legend(loc="upper right")


def plot_results(df, output_folder):
    blackboxes = ["DeepAR", "fcnet", "xgboost", "NAS"]
    optimizers_to_plot = [
        ["RS", "GP", "GP+prior", "TS", "CTS", "GCP", "GCP+prior"]
    ]

    fig, axes = plt.subplots(len(blackboxes), 1, figsize=(10, 12), sharex='row', sharey='row')
    for i, blackbox in enumerate(blackboxes):
        plot_optimizers(df, ax=axes[i], blackbox=blackbox, optimizers=optimizers_to_plot[0], legend=(i == 0))
    plt.savefig(Path(output_folder) / "Reproduce.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default="pytorch", help="Type of prior knowledge")
    parser.add_argument('--num_seeds', type=int, default=2, help="Number of random seeds")
    parser.add_argument('--num_evaluations', type=int, default=20, help="Number of evaluations per seed")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the results")
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help="Number of CPU workers to use")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting evaluations with the following parameters: {args}")

    main(
        prior=args.prior,
        num_seeds=args.num_seeds,
        num_evaluations=args.num_evaluations,
        output_folder=args.output_folder,
        num_workers=args.num_workers,
    )

