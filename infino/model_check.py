### Look at MCSE relative to StdDev "other_log_contribution_per_gene"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_stan_summary_metric(stan_summary, metric, parameter, title=None, figure_save_path=None):
    """
    :param stan_summary: DataFrame of the stansummary csv
    :param metric: one of ["R_hat", "MCSE", "N_Eff", 'StdDev']
    :param parameter: name of stan parameter to check, e.g. 'unknown_prop'
    :param title: title for figure
    :param figure_save_path: directory
    :return: plots the figure inline, optionally saves
    """
    f = plt.figure()
    sns.distplot(stan_summary[stan_summary.name.str.startswith(parameter)][metric])
    if title is None:
        plt.title('{} metric of convergence: {}'.format(metric, parameter))
    else:
        plt.title(title)

    plt.ylabel('Frequency')
    plt.xlabel(metric)

    if figure_save_path is not None:
        savefig(f, figure_save_path, dpi=300)


def analyze_mcse(stan_summary, parameter, title=None, figure_save_path=None):
    """
    E.g. [analyze_mcse(stan_summary, parameter) for parameter in parameters]
    :param stan_summary:
    :param parameter:
    :return:
    """
    parameter_df = stan_summary[stan_summary.name.str.startswith(parameter)]

    ratio = parameter_df['MCSE'] / parameter_df['StdDev']
    print(ratio.describe())

    f = plt.figure(figsize=(18,8))
    sns.regplot(np.array(range(len(ratio))), ratio, fit_reg=False)

    if title is None:
        plt.title('MCSE/StdDev ratio of {}'.format(parameter))
    else:
        plt.title(title)

    if figure_save_path is not None:
        savefig(f, figure_save_path, dpi=300)


def savefig(fig, *args, **kwargs):
    """
    Wrap figure.savefig defaulting to tight bounding box.
    From https://github.com/mwaskom/seaborn/blob/dfdd1126626f7ed0fe3737528edecb71346e9eb0/seaborn/axisgrid.py#L1840
    """
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(*args, **kwargs)