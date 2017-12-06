import numpy as np
import pandas as pd
import re

from . import config

from config import CELL_TYPES, ROLLUPS

def get_trace_columns(parameter, stan_summary):
    cols_we_want = stan_summary[stan_summary.name.str.startswith(parameter)].name.values
    trace_columns = [c.replace('[', '.').replace(']', '').replace(',', '.') for c in cols_we_want]
    return trace_columns


# From the traces, get `merged_samples`
def traces_to_dataset(trace_filenames_list, 
                      trace_columns, 
                     warmup, 
                      stan_summary,
                     rollups=ROLLUPS,                       
                     cell_types=CELL_TYPES, 
                     logging=True):
    
    all_traces_list = []
    
    cell_types = np.array(cell_types)
    
    for (i, f) in zip(range(len(trace_filenames_list)), trace_filenames_list):
        trace_df = pd.read_csv(f, comment='#', usecols=trace_columns)
        trace_df['trace_id'] = i
        trace_df['iter'] = trace_df.index
        all_traces_list.append(trace_df)

    all_traces_df = pd.concat(all_traces_list)
    
    # Munge all_traces_df 
    
    all_traces_df2 = pd.melt(all_traces_df, id_vars=['iter','trace_id'], value_name='estimate', var_name='variable')
    var_ids = all_traces_df2.variable.str.extract('sample2_x.(?P<sample_id>\d+).(?P<subset_id>\d+)')

    all_traces_df3= pd.concat([all_traces_df2, var_ids], axis=1) 
    all_traces_df3['subset_id'] = all_traces_df3['subset_id'].astype(int)
    all_traces_df3['sample_id'] = all_traces_df3['sample_id'].astype(int)

    sample2_xs = stan_summary[stan_summary['name'].str.startswith('sample2_x')]['Mean'].values.reshape(all_traces_df3.sample_id.max(), all_traces_df3.subset_id.max()) # (10,13) before

    ## Edit this - include the unknown prop? 
    mixture_estimates = pd.DataFrame(sample2_xs, columns=cell_types)
    
    subset_names = [re.sub(string=x, pattern='(.*)\[(.*)\]', repl='\\2') for x in mixture_estimates.columns]
    
    all_traces_df3['subset_name'] = all_traces_df3.subset_id.apply(lambda i: subset_names[i-1])
    
        
    # Warmup
    if logging:
        print("Pre-warmup")
        print(all_traces_df3.iter.describe()[['min', 'max']])
    
    all_traces_df3 = all_traces_df3.loc[all_traces_df3['iter']>=warmup,]
    all_traces_df3['iter'] -= warmup

    if logging:
        print("Post-warmup")
        print(all_traces_df3.iter.describe()[['min', 'max']])
    
    num_samples = all_traces_df3.iter.max() + 1
    
    # combine iteration numbers across traces -- i.e. line them up from 0 to 4000, not 4 versions of 0 to 1000
    #(all_traces_df3['trace_id']*1000 + all_traces_df3['iter']).hist()
    (all_traces_df3['trace_id']*num_samples + all_traces_df3['iter']).describe()[['min', 'max']]
    
    all_traces_df3['combined_iter_number'] = (all_traces_df3['trace_id']*num_samples + all_traces_df3['iter'])
    
    assert all_traces_df3.shape[0] / all_traces_df3.sample_id.max() / all_traces_df3.subset_id.max() / 4 == num_samples
    
    # Add rollup column
    
    all_traces_df3['rollup'] = all_traces_df3.subset_name.apply(lambda x: label_rollup(rollups, x))
    
    samples_rolledup = all_traces_df3.groupby(['sample_id', 'combined_iter_number', 'rollup']).estimate.sum().reset_index()
    
    cleaner_traces = all_traces_df3.copy()
    cleaner_traces['subset_name'] = cleaner_traces['subset_name'].str.replace('_', ' ')
    
    merged_samples_1 = cleaner_traces[['sample_id', 'combined_iter_number', 'subset_name', 'estimate']].copy()
    merged_samples_1['type'] = 'subset'
    merged_samples_2 = samples_rolledup.copy()
    merged_samples_2.columns = [c.replace('rollup', 'subset_name') for c in merged_samples_2.columns]
    merged_samples_2['type'] = 'rollup'
    merged_samples = pd.concat([merged_samples_1, merged_samples_2])
    
    return merged_samples

def label_rollup(rollups, x):
    for key in rollups.keys():
        if x in rollups[key]:
            return key
    return None
    