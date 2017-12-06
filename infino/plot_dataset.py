# Plots mcmc and violin plots from `merged_samples`
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from . import CELL_TYPES, ROLLUPS
from plot_settings import PlotSettings

class PlotDataset:
    def __init__(self,
                 key,
                  trace_samples_df,
                  cibersort_df,
                  cibersort_rollup_df,
                 plot_settings=None,
                 rollups=ROLLUPS
                 ):

        infino_vals = trace_samples_df[trace_samples_df['sample_id'] == key]
        cibersort_vals_base = cibersort_df[CELL_TYPES].iloc[key - 1]
        cibersort_vals_rolledup = cibersort_rollup_df.iloc[key - 1]

        infino_vals = infino_vals.copy()  # because going to moplodify
        infino_vals['supertype'] = infino_vals['subset_name'].apply(
            lambda x: 'CD4 T' if 'CD4' in x else 'CD8 T' if 'CD8' in x else 'UNKNOWN' if 'unknown' else 'B')

        map_row_to_ylevel = {}
        for k in rollups:
            map_row_to_ylevel[k] = 0
            for v, s in zip(rollups[k], range(1, len(rollups[k]) + 1)):
                map_row_to_ylevel[v.replace('_', ' ')] = s

        infino_vals['ylevel'] = infino_vals['subset_name'].apply(lambda x: map_row_to_ylevel[x])

        merged_grp = infino_vals

        # add cibersort points
        cb_base = pd.DataFrame(self.cibersort_vals_base).reset_index()
        cb_base.columns = ['SubSet', 'estimate']
        cb_base['type'] = 'subset'

        cb_rolledup = pd.DataFrame(self.cibersort_vals_rolledup).reset_index()
        cb_rolledup.columns = ['SubSet', 'estimate']
        cb_rolledup['type'] = 'rollup'

        cb = pd.concat([cb_base, cb_rolledup])
        cb.SubSet = cb.SubSet.str.replace('_', ' ')  # normalize names

        cb = cb.rename(columns={'estimate': 'cb'})

        merged_grp2 = pd.merge(merged_grp, cb, left_on='subset_name', right_on='SubSet', how='left')

        # turn ylevel into a categorical variable so that violinplot knows what to do with it
        merged_grp2['ylevel_str'] = merged_grp2['ylevel'].apply(lambda x: chr(65 + x))
        merged_grp2['ylevel_str'].unique()

        self.plot_dataset = merged_grp2

        self.plot_settings = PlotSettings() if plot_settings is None else plot_settings



    def plot_mcmc_areas(self, relative_to_groundtruth=False):
        """
        plot MCMC areas, perhaps relative_to_groundtruth if flag enabled
        feed in output of merge_datasets_for_plots(mixID)
        """
        # HERE THERE'S NO GT
        assert not relative_to_groundtruth

        estimate_var = 'estimate_rel_gt' if relative_to_groundtruth else 'estimate'
        gt_var = 'gt_rel_gt' if relative_to_groundtruth else 'gt'
        cb_var = 'cb_rel_gt' if relative_to_groundtruth else 'cb'
        label_cut_point = 0 if relative_to_groundtruth else .5

        with sns.plotting_context("talk"):
            with sns.axes_style("white", rc={"axes.facecolor": (0, 0, 0, 0)}):
                g = sns.FacetGrid(self.plot_dataset,
                                  row="ylevel",
                                  hue="subset_name",
                                  col="supertype",
                                  row_order=reversed(list(range(self.plot_dataset.ylevel.values.max() + 1))),
                                  hue_order=self.plot_settings.hue_order,
                                  aspect=15,
                                  size=.5,
                                  palette=self.plot_settings.built_palette,
                                  sharey=False  # important -- they don't share y ranges.
                                  )

                ## Draw the densities in a few steps
                # this is the shaded area
                g.map(sns.kdeplot,
                      estimate_var,
                      clip_on=False,
                      shade=True,
                      alpha=.8,
                      lw=2,
                      )

                # this is the dividing horizontal line
                g.map(plt.axhline, y=0, lw=2, clip_on=False, ls='dashed')

                ### Add label for each facet.

                def label(type_series, estimates, cut_point=.5, **kwargs):
                    """
                    type_series is a Series that corresponds to this facet. it will have values "subset" or "rollup"
                    kwargs is e.g.: {'color': (0.4918017777777778, 0.25275644444444445, 0.3333333333333333), 'label': 'CD4 Treg'}
                    use estimates to find median. put rollup label on left/right based on that
                    """
                    type_of_label = type_series.values[0]
                    color = kwargs['color']
                    label = kwargs['label']
                    estimate_median = estimates.median()
                    ax = plt.gca()  # map() changes current axis repeatedly
                    if type_of_label == 'rollup':
                        plot_on_right = (estimate_median <= cut_point)
                        ax.text(
                            1 if plot_on_right else 0,
                            .2,
                            label + " (sum)",  # label,
                            fontweight="bold",
                            color=color,
                            ha="right" if plot_on_right else "left",
                            va="center", transform=ax.transAxes,
                            fontsize='x-large',  # 15,
                            bbox=dict(facecolor='yellow', alpha=0.3)
                        )
                    else:
                        ax.text(1, .2,
                                label,
                                fontweight="bold",
                                color=color,
                                ha="right", va="center", transform=ax.transAxes
                                )

                g.map(label, "type_x", estimate_var, cut_point=label_cut_point)

                ### Overlay Cibersort and Ground Truth points at the right heights.

                def get_kde_intersection_yval(x0, kde_x, kde_y):
                    """
                    we want to find y value at which kde line
                    (defined by [kde_x, kde_y] point collection)
                    intersects x=x0
                    (kde_x, kde_y are numpy ndarrays)
                    """
                    if x0 in kde_x:
                        # the point actually is in the kde point definition!
                        return kde_y[np.where(kde_x == x0)][0]
                    elif not (kde_x.min() <= x0 <= kde_x.max()):
                        # out of bounds of the kde
                        return 0
                    else:
                        # need to interpolate
                        # find the two x values that most closely encircle x0
                        # then take average of their y values
                        # i.e. linear approximation
                        diffs = np.abs(kde_x - x0)
                        idxs = np.argsort(diffs)  # like argmin, but multiple outputs -- indexes for sorted order
                        argmins = idxs[:2]
                        return np.mean(kde_y[argmins])

                def plot_point(gt, scattercolor, legendlabel, plotline=True, linestyle='solid', linecolor='k', s=100,
                               zorder=10, **kwargs):
                    """
                    custom function to overlay ground truth and cibersort
                    method signature is: *args, **kwargs
                    make sure not to have any custom kwargs named "color" or "label"
                    those are passed in by default related to facet.. avoid
                    """

                    # get rid of "label" and "color" that seaborn passes in
                    # so we don't pass double kwargs to ax.scatter
                    kwargs.pop('label', None)
                    kwargs.pop('color', None)

                    ax = plt.gca()
                    # passed in a gt series
                    # all values are the same, since we did a left merge
                    # (there's only one gt value per facet)
                    xval = gt.values[0]

                    # get y value of kde at this xval
                    kde_line = ax.get_lines()[0].get_data()
                    ymax = get_kde_intersection_yval(xval, kde_line[0], kde_line[1])
                    # plot
                    ax.scatter([xval], [ymax],
                               s=s,
                               c=scattercolor,
                               zorder=zorder,
                               clip_on=False,  # means not clipped by axis limits (so we see the whole circle)
                               label=legendlabel,
                               **kwargs
                               )
                    if plotline:
                        ax.vlines(x=xval, ymin=0, ymax=ymax, linewidths=2, colors=linecolor, linestyles=linestyle)
                        # not axvline, because ymin,ymax are in axes coordinates for axvline, not in data coordinates

                # cibersort
                g.map(plot_point,
                      cb_var,
                      scattercolor=self.plot_settings.paired_colors[6],
                      linecolor=self.plot_settings.paired_colors[6],
                      legendlabel='Cibersort',
                      zorder=10,
                      alpha=.8
                      )

                ## Beautify the plot.

                # change x axis ranges
                if relative_to_groundtruth:
                    # compute sensible xrange -- round up to nearest .25 (i.e. use math.ceil() not round())
                    # we do this manually rather than setting sharex=True because want 0 to be centered (i.e. +/- same amount)
                    xrng = np.ceil(max(self.plot_dataset.estimate_rel_gt.abs().max(),
                                       self.plot_dataset.cb_rel_gt.abs().max()) * 4) / 4
                    g.set(
                        xlim=(-xrng - .02, xrng + .02),  # so label shows up
                        xticks=np.arange(-xrng, xrng + .01, .1),  # so final tick is included
                    )
                else:
                    g.set(xlim=(-0.01, 1.01))

                # change y axis ranges
                g.set(ylim=(0, None))  # seems to do the trick along with sharey=False

                # Disable overlap.
                # TODO: remove this?
                # Some `subplots_adjust` line is necessary. without this, nothing appears
                g.fig.subplots_adjust(hspace=0)

                # Remove axes details that don't play will with overlap
                g.set_titles("")
                # g.set_titles(col_template="{col_name}", row_template="")
                g.set(yticks=[], ylabel='')
                g.despine(bottom=True, left=True)

                # fix x axis
                g.set_xlabels(
                    'Mixture proportion relative to ground truth' if relative_to_groundtruth else 'Mixture proportion')

                # resize
                cur_size = g.fig.get_size_inches()
                increase_vertical = 7  # 4 # 3
                g.fig.set_size_inches(cur_size[0], cur_size[1] + increase_vertical)

                # legend
                handles, labels = g.fig.gca().get_legend_handles_labels()
                chosen_labels_idx = [
                    # labels.index('Ground Truth'),
                    labels.index('Cibersort')
                ]
                legend = g.fig.gca().legend([handles[i] for i in chosen_labels_idx],
                                            [labels[i] for i in chosen_labels_idx],
                                            loc='upper right',
                                            frameon=True,
                                            bbox_to_anchor=(0, -0.1, 1, 1),
                                            bbox_transform=g.fig.transFigure
                                            )
                # without bbox_to_anchor this gets applied to upper right of the last axis
                frame = legend.get_frame()
                frame.set_edgecolor(self.plot_settings.built_palette[0])
                frame.set_facecolor('white')

                # tighten
                g.fig.tight_layout()

                # done
                return g, g.fig




        