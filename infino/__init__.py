# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
from os import path

# TODO: These should be specified in a config or as sys env vars probably. putting them here for convenience for now.

CIBERSORT_DATA_DIR = path.join('..', 'data')
STAN_DATA_DIR = '/Users/eliza/data/modelcache_new/experiments/20171106_6.4otherbucket'

CELL_TYPES = ['B_CD5', 'B_Memory', 'B_Naive', 'CD4_Central_Memory',
       'CD4_Effector_Memory', 'CD4_Naive', 'CD4_Th1', 'CD4_Th17',
       'CD4_Th2', 'CD4_Treg', 'CD8_Central_Memory', 'CD8_Effector',
       'CD8_Naive', 'unknown_prop']

ROLLUPS = {
    'B': [c for c in CELL_TYPES if c.startswith('B_')],
    'CD4 T': [c for c in CELL_TYPES if c.startswith('CD4_')],
    'CD8 T': [c for c in CELL_TYPES if c.startswith('CD8_')],
    'Unknown': [c for c in CELL_TYPES if c.startswith('unknown_')]
}

CIBERSORT_CELL_TYPES = ['B_CD5', 'B_Memory', 'B_Naive', 'CD4_Central_Memory',
       'CD4_Effector_Memory', 'CD4_Naive', 'CD4_Th1', 'CD4_Th17',
       'CD4_Th2', 'CD4_Treg', 'CD8_Central_Memory', 'CD8_Effector',
       'CD8_Naive']

CIBERSORT_ROLLUPS = {
    'B': [c for c in CELL_TYPES if c.startswith('B_')],
    'CD4 T': [c for c in CELL_TYPES if c.startswith('CD4_')],
    'CD8 T': [c for c in CELL_TYPES if c.startswith('CD8_')]}

STAN_PARAMETER_NAMES = {'cell_types_prefix': 'sample2_x',
                        'unknown_prefix': 'unknown_prop'} # Used in get_trace_columns