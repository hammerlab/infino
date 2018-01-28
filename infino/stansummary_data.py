import numpy as np
import pandas as pd
import re

from . import CELL_TYPES, ROLLUPS, STAN_PARAMETER_NAMES

def get_trace_columns(parameters_list, stan_summary):
    """

    :param parameters_list:
    :param stan_summary:
    :return:
    """
    cols_we_want = []
    for parameter in parameters_list:
        cols_we_want.extend(stan_summary[stan_summary.name.str.startswith(parameter)].name.values)

    trace_columns = [c.replace('[', '.').replace(']', '').replace(',', '.') for c in cols_we_want]
    return trace_columns


def traces_to_dataset(trace_filenames_list,
                      stan_summary,
                      parameters=STAN_PARAMETER_NAMES,
                      warmup=0,
                      rollups=ROLLUPS,
                      cell_types=CELL_TYPES,
                      logging=True):
    """
    Combines all trace csv's and the stansummary csv for the parameters of interest

    :param trace_filenames_list:
    :param parameters_list:
    :param warmup:
    :param stan_summary: DataFrame
    :param rollups: dict of rollups mapping to cell_types
    :param cell_types: list of cell type names that corresponds to cibersort names
    :param logging: boolean
    :return: Returns DataFrame with columns: sample_id, combined_iter_number, subset_name, estimate, type
    """

    trace_columns = get_trace_columns(parameters.values(), stan_summary)

    all_traces_list = []
    
    cell_types = np.array(cell_types)
    
    for (i, f) in zip(range(len(trace_filenames_list)), trace_filenames_list):
        trace_df = pd.read_csv(f, comment='#', usecols=trace_columns)
        trace_df['trace_id'] = i
        trace_df['iter'] = trace_df.index
        all_traces_list.append(trace_df)

    all_traces_df = pd.concat(all_traces_list)
    
    # Munge all_traces_df 
    cell_types_prefix = parameters['cell_types_prefix']
    unknown_prefix = parameters['unknown_prefix']

    all_traces_melted = pd.melt(all_traces_df, id_vars=['iter','trace_id'], value_name='estimate', var_name='variable')

    cell_traces_df = all_traces_melted[all_traces_melted.variable.str.startswith(cell_types_prefix)]
    unknown_traces_df = all_traces_melted[all_traces_melted.variable.str.startswith(unknown_prefix)]

    cell_var_ids = cell_traces_df.variable.str.extract(cell_types_prefix + '.(?P<sample_id>\d+).(?P<subset_id>\d+)')
    #print(cell_var_ids)
    cell_traces_df3 = pd.concat([cell_traces_df, cell_var_ids], axis=1)
    cell_traces_df3['subset_id'] = cell_traces_df3['subset_id'].astype(int)
    cell_traces_df3['sample_id'] = cell_traces_df3['sample_id'].astype(int)

    sample2_xs = stan_summary[stan_summary['name'].str.startswith(cell_types_prefix)]['Mean'].values.reshape(cell_traces_df3.sample_id.max(), cell_traces_df3.subset_id.max())

    from . import CIBERSORT_CELL_TYPES
    mixture_estimates = pd.DataFrame(sample2_xs, columns=np.array(CIBERSORT_CELL_TYPES))
    
    subset_names = [re.sub(string=x, pattern='(.*)\[(.*)\]', repl='\\2') for x in mixture_estimates.columns]

    cell_traces_df3['subset_name'] = cell_traces_df3.subset_id.apply(lambda x: subset_names[x-1])

    # Drop the warmup samples
    if logging:
        print("Pre-warmup")
        print(cell_traces_df3.iter.describe()[['min', 'max']])
    
    cell_traces_post_warmup = cell_traces_df3.loc[cell_traces_df3['iter'] >= warmup,]
    cell_traces_post_warmup['iter'] -= warmup

    if logging:
        print("Post-warmup")
        print(cell_traces_post_warmup.iter.describe()[['min', 'max']])
    
    num_samples = cell_traces_post_warmup.iter.max() + 1
    
    # combine iteration numbers across traces -- i.e. line them up from 0 to 4000, not 4 versions of 0 to 1000
    (cell_traces_post_warmup['trace_id']*num_samples + cell_traces_post_warmup['iter']).describe()[['min', 'max']]
    
    cell_traces_post_warmup['combined_iter_number'] = (cell_traces_post_warmup['trace_id']*num_samples + cell_traces_post_warmup['iter'])
    
    assert cell_traces_post_warmup.shape[0] / cell_traces_post_warmup.sample_id.max() / cell_traces_post_warmup.subset_id.max() / 4 == num_samples
    
    # Add rollup column
    
    cell_traces_post_warmup['rollup'] = cell_traces_post_warmup.subset_name.apply(lambda x: label_rollup(rollups, x))
    
    samples_rolledup = cell_traces_post_warmup.groupby(['sample_id', 'combined_iter_number', 'rollup']).estimate.sum().reset_index()
    
    cleaner_traces = cell_traces_post_warmup.copy()
    cleaner_traces['subset_name'] = cleaner_traces['subset_name'].str.replace('_', ' ')
    
    merged_samples_1 = cleaner_traces[['sample_id', 'combined_iter_number', 'subset_name', 'estimate']].copy()
    merged_samples_1['type'] = 'subset'
    merged_samples_2 = samples_rolledup.copy()
    merged_samples_2.columns = [c.replace('rollup', 'subset_name') for c in merged_samples_2.columns]
    merged_samples_2['type'] = 'rollup'
    merged_samples = pd.concat([merged_samples_1, merged_samples_2])

    ### unknown prop df

    unknown_var_ids = unknown_traces_df.variable.str.extract('unknown_prop' + '.(?P<sample_id>\d+)')
    df3 = pd.concat([unknown_traces_df, unknown_var_ids], axis=1)

    unknown_traces_post_warmup = df3.loc[df3['iter'] >= warmup,]
    unknown_traces_post_warmup['iter'] -= warmup

    unknown_traces_post_warmup['combined_iter_number'] = unknown_traces_post_warmup['iter'] + unknown_traces_post_warmup['trace_id'] * num_samples
    unknown_traces_post_warmup.loc[:, 'subset_name'] = 'Unknown'
    unknown_traces_post_warmup.loc[:, 'type'] = 'Unknown'
    unknown_traces_post_warmup.loc[:, 'sample_id'] = unknown_traces_post_warmup['sample_id'].astype(int)
    unknowns_merged_samples = unknown_traces_post_warmup[['sample_id', 'combined_iter_number', 'subset_name', 'estimate', 'type']]

    all_merged_samples = pd.concat([merged_samples, unknowns_merged_samples])

    return all_merged_samples

def label_rollup(rollups, x):
    for key in rollups.keys():
        if x in rollups[key]:
            return key
    return None
    