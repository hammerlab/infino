import pandas as pd
from sys import path
from . import CIBERSORT_DATA_DIR
from . import CIBERSORT_CELL_TYPES, CIBERSORT_ROLLUPS

def get_cibersort_dfs(cibersort_filepath, rollups=CIBERSORT_ROLLUPS):
    cibersort_df = pd.read_csv(cibersort_filepath, sep='\t')  # , index_col=0)
    cibersort_df.drop('Unnamed: 0', axis=1, inplace=True)  # this is index from sub-dfs. not monotonically increasing
    cibersort_df[['cohort', 'metric', 'processing', 'cutname']].drop_duplicates()

    rollup_sums = {}

    for key in rollups:
        if key is not 'Unknown':
            rollup_sums[key] = cibersort_df[rollups[key]].sum(axis=1)

    rollup_sums_df = pd.DataFrame(rollup_sums)

    return cibersort_df, rollup_sums_df


def get_cibersort_df(cibersort_filepath):
    cibersort_df = pd.read_csv(cibersort_filepath, sep='\t') #, index_col=0)
    cibersort_df.drop('Unnamed: 0', axis=1, inplace=True) # this is index from sub-dfs. not monotonically increasing
    cibersort_df[['cohort', 'metric', 'processing', 'cutname']].drop_duplicates()
   # cibersort_results = cibersort_results.loc[(
   # (cibersort_results['cohort'] == cohort_name_cib) & (cibersort_results['metric'] == metric) & (
   # cibersort_results['processing'] == processing) & (cibersort_results['cutname'] == cutname))]

    return cibersort_df


def get_cibersort_rollups_df(cibersort_df, rollups=CIBERSORT_ROLLUPS):
    rollup_sums = {}

    for key in rollups:
        rollup_sums[key] = cibersort_df[rollups[key]].sum(axis=1)

    rollup_sums_df = pd.DataFrame(rollup_sums)
    return rollup_sums_df

    
def get_cibersort_celltype_names(cibersort_classes_file='cohort_newbladder.cibersort.input.classes.datatype_est_counts.txt'):
    cibersort_classes = pd.read_csv(path.join(CIBERSORT_DATA_DIR, cibersort_classes_file), sep='\t', header=None)
    celltype_names = cibersort_classes[0].values
    return celltype_names
    

    

