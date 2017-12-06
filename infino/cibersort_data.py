import pandas as pd

def get_cibersort_df(cibersort_filepath):
    cibersort_df = pd.read_csv(cibersort_filepath, sep='\t') #, index_col=0)
    cibersort_df.drop('Unnamed: 0', axis=1, inplace=True) # this is index from sub-dfs. not monotonically increasing
    cibersort_df[['cohort', 'metric', 'processing', 'cutname']].drop_duplicates()
    #cibersort_df = cibersort_df.loc[((cibersort_df['cohort'] == cohort_name_cib) &  (cibersort_df['metric'] == metric) &  (cibersort_df['processing'] == processing) &  (cibersort_df['cutname'] == cutname))]
    
    #cibersort_df.reset_index(drop=True, inplace=True)
    
    return cibersort_df


def get_cibersort_rollups_df(cibersort_df, rollups):
    rollup_sums = {}
    
    for key in rollups:
        rollup_sums[key] = cibersort_df[rollups[key]].sum(axis=1)

    rollup_sums_df = pd.DataFrame(rollup_sums)
    return rollup_sums_df

    
def get_cibersort_celltype_names(cibersort_classes_filepath='cohort_newbladder.cibersort.input.classes.datatype_est_counts.txt'):
    cibersort_classes = pd.read_csv(cibersort_classes_filepath, sep='\t', header=None)
    celltype_names = cibersort_classes[0].values
    return celltype_names
    

    

