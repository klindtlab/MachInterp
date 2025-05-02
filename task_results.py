# Import requirements
import matplotlib.pyplot as plt
import numpy as np


def plotResults(results: dict, all_metrics: dict, codes: dict) -> None:
    
    """
    Plot MIS, OOO, and cross-MIS results for all metrics.

    Args:
        results (dict): Dictionary containing accuracy results from different tasks.
        all_metrics (dict): Dictionary with metric names as keys and metric objects (with `.num_scores`) as values.
        codes (dict): Dictionary mapping dataset keys to display labels for plotting.

    Returns:
        None
    """
    
    num_plots=2 # Num of plot types (line, boxplot)
    num_layers=0 # Total num layers across all metrics
    for item in all_metrics.items():
        num_layers+=item[1].num_scores
    plt.figure(figsize=(num_plots*4, num_layers*4))
    count=1
    for item in all_metrics.items():
        layers = item[1].num_scores
        name = item[0]
        
        # Line plot for dreamsim, other 1 layered metrics
        if layers == 1:
            plt.subplot(num_layers,num_plots,count)
            count+=1
            for key, result in results.items():
                if 'quantiles' in result.keys(): # on MIS task
                    plt.errorbar(result['quantiles'],
                                 result[f'accuracy_{name}'].mean(0),
                                 result[f'accuracy_{name}'].std(0) / np.sqrt(result[f'accuracy_{name}'].shape[0]),
                                 label=key)
                    plt.xlabel('quantile')
                    plt.title(f'MIS - {name}')
                elif (len(result[f'accuracy_{name}'].shape) == 2): # on OOO task
                    plt.errorbar(np.log2(result['ks']),
                                 result[f'accuracy_{name}'].mean(0),
                                 result[f'accuracy_{name}'].std(0) / np.sqrt(result[f'accuracy_{name}'].shape[0]),
                                 label=key)
                    plt.xlabel('K')
                    plt.title(f'OOO - {name}')
                else: # on crossMIS task
                    plt.errorbar(np.log2(result['ks']),
                                 result[f'accuracy_{name}'].mean((0, 1)),
                                 result[f'accuracy_{name}'].std((0, 1)) / result[f'accuracy_{name}'].shape[0],
                                 label=key,)
                    plt.xlabel('K')
                    plt.title(f'Cross MIS - {name}')
                    plt.xticks(np.log2(result['ks']), 
                               result['ks'])
                plt.ylabel('accuracy')
                plt.legend()
                plt.grid()
                    

            # Boxplot for dreamsim, other 1 layered metrics
            if (len(result[f'accuracy_{name}'].shape) == 2): #on OOO, MIS
                plt.subplot(num_layers, num_plots, count)
                count+=1
                plt.boxplot([result[f'accuracy_{name}'].mean(1) for result in results.values()])
                plt.xticks(np.arange(1, len(codes) + 1),list(codes.keys()))
                plt.title(f'Boxplot - {name}')
            else: 
                plt.subplot(num_layers, num_plots, count) #on cross MIS
                count+=1
                plt.boxplot([result[f'accuracy_{name}'].mean((1, 2)) for result in results.values()])
                plt.xticks(np.arange(1, len(codes) + 1),list(codes.keys()))
                plt.title(f'Boxplot - {name}')
            plt.ylabel('accuracy')
            plt.grid()

        # Line plot for LPIPS, other multi-layered metrics
        elif layers > 1: 
            for i in range(layers):
                # First layer (which has a different naming convention)
                if i==0:
                    plt.subplot(num_layers,num_plots,count)
                    count+=1
                    for key, result in results.items():
                        if 'quantiles' in result.keys(): # on MIS task
                            plt.errorbar(
                            result['quantiles'],
                            result[f'accuracy_{name}'].mean(0),
                            result[f'accuracy_{name}'].std(0) / np.sqrt(result[f'accuracy_{name}'].shape[0]),
                            label=key)
                            plt.xlabel('quantile')
                            plt.title(f'MIS - {name}')
                        elif (len(result[f'accuracy_{name}'].shape) == 2): # on OOO task
                            plt.errorbar(np.log2(result['ks']),
                                         result[f'accuracy_{name}'].mean(0),
                                         result[f'accuracy_{name}'].std(0) / np.sqrt(result[f'accuracy_{name}'].shape[0]),
                                         label=key)
                            plt.xlabel('K')
                            plt.title(f'OOO - {name}')
                        else: # on crossMIS task
                            plt.errorbar(np.log2(result['ks']),
                                         result[f'accuracy_{name}'].mean((0, 1)),
                                         result[f'accuracy_{name}'].std((0, 1)) / result[f'accuracy_{name}'].shape[0],
                                         label=key,)
                            plt.xlabel('K')
                            plt.title(f'Cross MIS - {name}')
                            plt.xticks(np.log2(result['ks']), result['ks'])        
                        plt.ylabel('accuracy')
                        plt.legend()
                        plt.grid()

                    # Boxplot for LPIPS, other multi-layered metrics
                    if (len(result[f'accuracy_{name}'].shape) == 2): # on OOO, MIS
                        plt.subplot(num_layers, num_plots, count)
                        count+=1
                        plt.boxplot([result[f'accuracy_{name}'].mean(1) for result in results.values()])
                        plt.xticks(np.arange(1, len(codes) + 1), list(codes.keys()))
                        plt.title(f'Boxplot - {name}')
                    else: 
                        plt.subplot(num_layers, num_plots, count) # on crossMIS
                        count+=1
                        plt.boxplot([result[f'accuracy_{name}'].mean((1, 2)) for result in results.values()])
                        plt.xticks(np.arange(1, len(codes) + 1), list(codes.keys()))
                        plt.title(f'Boxplot - {name}')
                    plt.ylabel('accuracy')
                    plt.grid()
                # Layers 2+
                else:
                    plt.subplot(num_layers,num_plots,count)
                    count+=1
                    for key, result in results.items():
                        if 'quantiles' in result.keys():
                            plt.errorbar(result['quantiles'],
                                         result[f'accuracy_{name}_{i}'].mean(0),
                                         result[f'accuracy_{name}_{i}'].std(0) / np.sqrt(result[f'accuracy_{name}_{i}'].shape[0]),
                                         label=key)
                            plt.xlabel('quantile')
                            plt.title(f'MIS - {name}_{i}')
                        elif (len(result[f'accuracy_{name}_{i}'].shape) == 2): # on OOO task
                            plt.errorbar(np.log2(result['ks']), 
                                         result[f'accuracy_{name}_{i}'].mean(0), 
                                         result[f'accuracy_{name}_{i}'].std(0) / np.sqrt(result[f'accuracy_{name}_{i}'].shape[0]),
                                         label=key)
                            plt.title(f'OOO - {name}_{i}')
                        else:
                            plt.errorbar(np.log2(result['ks']),
                                         result[f'accuracy_{name}_{i}'].mean((0, 1)),
                                         result[f'accuracy_{name}_{i}'].std((0, 1)) / result[f'accuracy_{name}_{i}'].shape[0],
                                         label=key)
                            plt.xlabel('K')
                            plt.title(f'Cross MIS - {name}_{i}')
                            plt.xticks(np.log2(result['ks']), result['ks'])        
                        plt.ylabel('accuracy')
                        plt.legend()
                        plt.grid()

                    # Boxplots for layers 2+
                    if (len(result[f'accuracy_{name}_{i}'].shape) == 2): # OOO, MIS
                        plt.subplot(num_layers, num_plots, count)
                        count+=1
                        plt.boxplot([result[f'accuracy_{name}_{i}'].mean(1) for result in results.values()])
                        plt.xticks(np.arange(1, len(codes) + 1),list(codes.keys()))
                        plt.title(f'Boxplot - {name}_{i}')
                    else: #crossMIS
                        plt.subplot(num_layers, num_plots, count)
                        count+=1
                        plt.boxplot([result[f'accuracy_{name}_{i}'].mean((1, 2)) for result in results.values()])
                        plt.xticks(np.arange(1, len(codes) + 1),list(codes.keys()))
                        plt.title(f'Boxplot - {name}_{i}')
                    plt.ylabel('accuracy')
                    plt.grid()
    plt.tight_layout()