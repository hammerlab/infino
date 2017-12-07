### Look at MCSE relative to StdDev "other_log_contribution_per_gene"

import matplotlib.pyplot as plt
import seaborn as sns


def plot_stan_summary_metric(stan_summary, metric, parameter, title=None, figure_save_path=None):
    """

    :param stan_summary:
    :param metric:
    :param parameter:
    :param title:
    :param figure_save_path:
    :return:
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


def analyze_mcse(stan_summary, parameter_name):
    """
    E.g. [analyze_mcse(stan_summary, parameter) for parameter in parameters]
    :param stan_summary:
    :param parameter_name:
    :return:
    """
    parameter_df = stan_summary[stan_summary.name.str.startswith(parameter_name)]

    ratio = parameter_df['MCSE'] / parameter_df['StdDev']
    print(ratio.describe())

    f = plt.figure(figsize=(18,8))
    plt.scatter(x=range(len(ratio)), y=ratio, s=1)
    plt.title("%s: Ratio of MCSE to posterior standard deviation" % parameter_name)


def savefig(fig, *args, **kwargs):
    """
    Wrap figure.savefig defaulting to tight bounding box.
    From https://github.com/mwaskom/seaborn/blob/dfdd1126626f7ed0fe3737528edecb71346e9eb0/seaborn/axisgrid.py#L1840
    """
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(*args, **kwargs)