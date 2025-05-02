import matplotlib.pyplot as plt
import numpy as np


def plotResults(results: dict, all_metrics: dict, codes: dict) -> None:
    """
    Plot MIS, OOO, and cross-MIS task results for all metrics.

    Args:
        results (dict): Dictionary containing task results.
        all_metrics (dict): Dictionary with metric names as keys and metric objects as values.
        codes (dict): Dictionary mapping dataset keys to display labels for plotting.

    Returns:
        None
    """

    num_plots = 2  # Num of types of plots (e.g., line and boxplot)
    num_layers = sum(metric.num_scores for metric in all_metrics.values()) # Total num layers across all metrics

    fig, axes = plt.subplots(num_layers, num_plots, figsize=(num_plots * 4, num_layers * 4))
    axes = axes.reshape(-1, num_plots)  # Ensure it's a 2D array even for 1 row

    ax_idx = 0  # Keeps track of the current row in the axes grid

    for name, metric in all_metrics.items():
        layers = metric.num_scores

        for i in range(layers):
            acc_key = f'accuracy_{name}' if i == 0 else f'accuracy_{name}_{i}' # Handle naming conventions for layer 1 v layers 2+

            # Line plot
            ax_line = axes[ax_idx, 0]
            for key, result in results.items():
                acc = result[acc_key]
                if 'quantiles' in result:
                    x = result['quantiles']
                    y = acc.mean(0)
                    yerr = acc.std(0) / np.sqrt(acc.shape[0])
                    ax_line.set_xlabel("quantile")
                    title = f"MIS - {name}" if i == 0 else f"MIS - {name}_{i}"
                elif len(acc.shape) == 2:  # OOO
                    x = np.log2(result['ks'])
                    y = acc.mean(0)
                    yerr = acc.std(0) / np.sqrt(acc.shape[0])
                    ax_line.set_xlabel("K")
                    title = f"OOO - {name}" if i == 0 else f"OOO - {name}_{i}"
                else:  # cross-MIS
                    x = np.log2(result['ks'])
                    y = acc.mean((0, 1))
                    yerr = acc.std((0, 1)) / acc.shape[0]
                    ax_line.set_xlabel("K")
                    ax_line.set_xticks(np.log2(result['ks']))
                    ax_line.set_xticklabels(result['ks'])
                    title = f"Cross MIS - {name}" if i == 0 else f"Cross MIS - {name}_{i}"

                ax_line.errorbar(x, y, yerr=yerr, label=key)
                ax_line.set_ylabel("accuracy")
                ax_line.set_title(title)
                ax_line.legend()
                ax_line.grid()

            # Boxplot
            ax_box = axes[ax_idx, 1]
            if len(acc.shape) == 2:
                data_for_box = [result[acc_key].mean(1) for result in results.values()] # OOO, MIS
            else:
                data_for_box = [result[acc_key].mean((1, 2)) for result in results.values()] # cross MIS

            ax_box.boxplot(data_for_box)
            ax_box.set_xticks(np.arange(1, len(codes) + 1))
            ax_box.set_xticklabels(list(codes.keys()))
            ax_box.set_ylabel("accuracy")
            ax_box.set_title(f"Boxplot - {name}" if i == 0 else f"Boxplot - {name}_{i}")
            ax_box.grid()

            ax_idx += 1

    plt.tight_layout()
    plt.show()
