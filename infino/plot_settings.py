import seaborn as sns
import numpy as np

from . import ROLLUPS

class PlotSettings:
    def __init__(self, rollups=ROLLUPS):
        paired_colors = sns.color_palette("Paired_r", 12)

        # colors for ground truth and cibersort overlays
        sns.palplot([paired_colors[4], paired_colors[6]])

        hue_order = []
        built_palette = []
        for r, start in zip(list(rollups), range(0, 3)):
            hue_order.extend(rollups[r])
            hue_order.append(r)
            full_pal = sns.cubehelix_palette(len(rollups[r]) + 3,
                                             start=start,
                                             rot=-.25,
                                             light=.7)
            # first #subtypes colors
            built_palette.extend(full_pal[:len(rollups[r])])
            # then a darker color (but not the darkest)
            built_palette.append(full_pal[-2])
        hue_order = [h.replace('_', ' ') for h in hue_order]
        sns.palplot(built_palette)

        self.built_palette = built_palette
        self.hue_order = hue_order
        self.paired_colors = paired_colors

        color_lightening_coeff = 1.2
        built_pal_lighter2 = [(np.array(i) * color_lightening_coeff).clip(0, 1) for i in built_palette]
        sns.palplot(built_pal_lighter2)