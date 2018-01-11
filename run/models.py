from fnmatch import fnmatch
import ntpath
import re
import patsy
import seaborn as sns
import pandas as pd
import numpy as np
from seed import seed
from data import prep_annotated_data, prep_filename_metadata
import os
import logging
from stancache.stancache import cached, cached_stan_fit, cached_stan_file
import cache
from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)

def filter_stan_summary(stan_fit, pars):
    fitsum = stan_fit.summary(pars=pars)
    res = pd.DataFrame(fitsum['summary'], columns=fitsum['summary_colnames'], index=fitsum['summary_rownames'])
    return res.loc[:,['mean','se_mean','sd','2.5%','50%','97.5%','Rhat']]


def print_stan_summary(stan_fit, pars):
    print(filter_stan_summary(stan_fit=stan_fit, pars=pars).to_string())


def plot_stan_summary(stan_fit, pars, metric='Rhat'):
    df = filter_stan_summary(stan_fit=stan_fit, pars=pars)
    sns.distplot(df[metric])


def extract_theta_summary(stan_fit, par='theta', colnames=['B_cell', 'T_cell'], gene_id='new_gene_id'):
    theta = stan_fit.extract(par)[par]
    dflist = list()
    for gene in np.arange(theta.shape[1]):
        part_df = pd.DataFrame(theta[:,gene,:], columns=colnames)
        part_df[gene_id] = gene+1
        part_df.reset_index(inplace=True)
        part_df.rename(columns={'index': 'iter'}, inplace=True)
        dflist.append(part_df)
    return pd.concat(dflist)


def prep_theta_summary(stan_fit, sample_df,
                       gene_id='new_gene_id',
                       gene_cat='new_gene_cat',
                       colnames=['B_cell', 'T_cell'],
                       expose_group=None,
                       **kwargs):
    theta_df = extract_theta_summary(stan_fit, gene_id=gene_id, colnames=colnames, **kwargs)
    gene_cat_map = sample_df.drop_duplicates(subset=[gene_id, gene_cat]).loc[:,[gene_id, gene_cat]]
    theta_df = pd.merge(theta_df, gene_cat_map, on=gene_id, how='left')
    theta_ldf = pd.melt(theta_df,
                    value_vars=colnames,
                    value_name='value',
                    id_vars=['iter', gene_id, gene_cat])
    
    ## summarize difference from mean among all items per iteration
    theta_ldf['mean_per_gene'] = theta_ldf.groupby([gene_id, 'iter'])['value'].transform(np.mean)
    theta_ldf['diff_from_mean'] = theta_ldf['value'] - theta_ldf['mean_per_gene']
    ## summarize mean per variable*gene over all iterations
    theta_ldf['mean_value'] = theta_ldf.groupby([gene_id, 'variable'])['value'].transform(np.mean)
    theta_ldf['mean_diff'] = theta_ldf.groupby([gene_id, 'variable'])['diff_from_mean'].transform(np.mean)
    theta_ldf['mean_abs_diff'] = abs(theta_ldf['mean_diff'])
    ## rank genes (by diff & value) within cell type
    theta_ldf_rank = theta_ldf.drop_duplicates(subset=[gene_id,'variable']).copy()
    theta_ldf_rank['mean_diff_rank'] = theta_ldf_rank.groupby(['variable'])['mean_diff'].rank(ascending=False) 
    theta_ldf_rank['mean_abs_diff_rank'] = theta_ldf_rank.groupby(['variable'])['mean_abs_diff'].rank(ascending=False) 
    theta_ldf_rank['mean_value_rank'] = theta_ldf_rank.groupby(['variable'])['mean_value'].rank(ascending=False)
    theta_ldf_rank['mean_value_arank'] = theta_ldf_rank.groupby(['variable'])['mean_value'].rank(ascending=True)
    theta_ldf_rank = theta_ldf_rank.loc[:, ['mean_diff_rank', 'mean_abs_diff_rank', 'mean_value_rank', gene_id, 'variable']]
    theta_ldf = pd.merge(theta_ldf, theta_ldf_rank, on=[gene_id, 'variable'])
    ## expose on set of cell-level data at group level, so this can be sorted on
    if not expose_group:
        expose_group = colnames[0]
    expose_data = theta_ldf.loc[theta_ldf['variable'] == expose_group, :].drop_duplicates(subset=gene_id)
    expose_fields = [field for field in list(expose_data.columns) 
                     if is_field_unique_by_group(expose_data, field_col=field, group_col=gene_id)]
    expose_fields = [field for field in expose_fields 
                     if field.startswith('mean') 
                     or field.startswith('diff') 
                     or field == gene_id]
    expose_data = expose_data.loc[:, expose_fields]
    theta_ldf = pd.merge(theta_ldf, expose_data, on=gene_id, how='left', suffixes=['','_{}'.format(expose_group)])
    return theta_ldf

    
def compute_rank_within_group(df, group_by, rank_of):
    df['{}_rank'.format(rank_of)] = df.groupby(group_by)[rank_of].rank()
    return df

def gene_list_from_sample_df(sample_df):
    """
    extracts gene list from a sample_df produced by prep_sample_df
    """
    return sample_df.gene_name.unique().tolist()


def prep_sample_df(df=None, 
                    sample_n=None,
                    drop_zero_values=False,
                    y_col='est_counts',
                    sample_gene_list=None):
    """
    Generates training dataframe from [y_col]. 
    If [sample_n] is set, only that many genes will be sampled. Otherwise all genes are returned.
    [sample_gene_list] overrides [sample_n]. If [sample_gene_list] is set, only those genes will be included in the output.
    See companion method gene_list_from_sample_df to extract a gene list that can be inserted here.
    """
    if df is None:
        df = cached(prep_annotated_data)
    # optionally limit to genes with at least one observed count
    if drop_zero_values:
        all_genes = df.loc[df['est_counts']==0,:].drop_duplicates(subset='gene_name').loc[:,'gene_name']
    else:
        all_genes = df.drop_duplicates(subset='gene_name').loc[:,'gene_name']
    # optionally sample from genes
    if sample_gene_list:
        sampled_genes = all_genes[all_genes.isin(sample_gene_list)]
    elif sample_n:
        sampled_genes = all_genes.sample(n=sample_n)
    else:
        sampled_genes = all_genes
    # sample_df & re-generate id variables
    sample_df = pd.merge(df, pd.DataFrame(sampled_genes), on='gene_name', how='inner')
    sample_df.sort_values(['gene_id','sample_id'], inplace=True)
    sample_df['new_gene_cat'] = sample_df['gene_name'].astype('category')
    sample_df['new_gene_id'] = sample_df['new_gene_cat'].cat.codes+1
    sample_df['new_sample_cat'] = sample_df['sample_id'].astype('category')
    sample_df['new_sample_id'] = sample_df['new_sample_cat'].cat.codes+1
    sample_df.reset_index(inplace=True)
    return sample_df


def split_sample_df(sample_df=None, test_sample_n=1, train_sample_n=None, **kwargs):
    if sample_df is None:
        sample_df = cached(prep_sample_df, **kwargs)
    # split sample_df into training & test sets
    all_samples = sample_df.drop_duplicates(subset='sample_id')['sample_id']
    if test_sample_n and test_sample_n > 0:
        test_samples = all_samples.sample(n=test_sample_n)
    else:
        raise ValueError('No test samples identified.')
    # split sample df into two datasets
    training_df = sample_df[sample_df['sample_id'].apply(lambda x: x not in test_samples)].copy()
    test_df = sample_df[sample_df['sample_id'].apply(lambda x: x in test_samples)].copy()
    if train_sample_n:
        train_samples = training_df.drop_duplicates(subset='sample_id')['sample_id']
        train_samples = train_samples.sample(n=train_sample_n)
        training_df = training_df[training_df['sample_id'].apply(lambda x: x in train_samples)]
    for df in [training_df, test_df]:
        df.sort_values(['gene_id','sample_id'], inplace=True)
        df['new_sample_cat'] = df['sample_id'].astype('category')
        df['new_sample_id'] = df['new_sample_cat'].cat.codes+1
    return (training_df, test_df)


def prep_yrep_summary(stan_fit, sample_df, par='y_rep', value_name='pp_est_counts', sample_kwds=None,
                     filter_genes=None):
    yrep = stan_fit.extract(par)[par]
    if filter_genes is not None:
        # if filtering on genes, identify which yrep indices correspond to those genes
        # we ideally want to restrict our sample to these as early as possible
        yrep_indices = list(sample_df.loc[sample_df['new_gene_cat'].isin(filter_genes),'index'].values)
        yrep = yrep[:,yrep_indices]
    else:
        yrep_indices = None
    yrep_df = pd.DataFrame(yrep, columns=yrep_indices)
    yrep_df.reset_index(inplace=True)
    yrep_df.rename(columns={'index': 'iter'}, inplace=True)
    if sample_kwds:
        yrep_df = yrep_df.sample(**sample_kwds)
    yrep_ldf = pd.melt(yrep_df, id_vars='iter', value_name=value_name, var_name='index')
    try:
        sample_df.reset_index(inplace=True)
    except:
        pass
    yrep_ldf = pd.merge(yrep_ldf,
                        sample_df,
                        suffixes=['.ppcheck',''],
                        on='index')
    return yrep_ldf


def prep_omega_summary(stan_fit, stan_data, gene_id='cell_type', par='Omega', return_summary=True):
    omega_df = extract_theta_summary(stan_fit=stan_fit,
                                          colnames=list(stan_data['x'].columns),
                                          gene_id=gene_id,
                                          par=par)
    omega_df[gene_id] = omega_df[gene_id].apply(lambda x: list(stan_data['x'].columns)[x-1])
    omega_summary = omega_df.groupby(gene_id).apply(lambda x: np.mean(x))
    if return_summary:
        return omega_summary
    else:
        return omega_df
        

    
def prep_theta_mu_summary(stan_fit, stan_data, par='theta_mu'):
    mu_ex = stan_fit.extract(par)[par]
    mu_df = pd.DataFrame(mu_ex, columns=list(stan_data['x'].columns))
    mu_df.reset_index(inplace=True)
    mu_df.rename(columns = {'index': 'iter'}, inplace=True)
    mu_ldf = pd.melt(mu_df, id_vars='iter', value_vars=list(stan_data['x'].columns))
    return mu_ldf


def patsy_helper_nointercept(df, formula):
    model_frame = patsy.dmatrix('{} - 1'.format(formula), data=df, return_type='dataframe')
    if 'Intercept' in model_frame.columns:
        model_frame.drop('Intercept', axis=1, inplace=True)
    return model_frame


def is_field_unique_by_group(df, field_col, group_col):
    ''' Determine if field is constant by group in df
    '''
    def num_unique(x):
        return len(pd.unique(x))
    num_distinct = df.groupby(group_col)[field_col].agg(num_unique)
    return all(num_distinct == 1)

    
def prep_cell_data(df, selected_col, selected_values, 
                  fields = None):
    ''' cell-type level features according to 'selected-on' groups. 
        where selected-on is a dict mapping names of elements to items
        
        Parameters
        ------------
            df: dataframe containing cell-level features
            selected_col: field containing cell identifier
            selected_values: values of field containing cell identifier,
                        in the order in which they are given to Stan
            fields: (optional) list of fields containing cell-level features
                    if fields is None, queries `data.get_file_metadata` to 
                    get list of fields
    '''
    if not fields:
        fields = list(prep_filename_metadata().columns)
        # exclude filename & SubSet
        fields = [field for field in fields if field not in ['filename', 'SubSet']]
        # filter to those present in df
        fields = [field for field in fields if field in list(df.columns)]
    
    # filter to those unique by group in df
    fields = [field for field in fields if is_field_unique_by_group(df=df, field_col=field, group_col=selected_col)]
    
    # clean selected values of patsy-style naming, if it exists
    selected_vals = [re.sub('{}\[(.*)\]'.format(selected_col), '\\1', val) for val in selected_values]
    
    # get subset of data for selected key-value pairs
    selected_df = df.drop_duplicates(subset=selected_col)
    selected_df.set_index(selected_col, inplace=True)
    
    # return fields ordered to match order of keys given
    return selected_df.loc[:, fields]


def prep_stan_data(sample_df, by='cell_type', cell_features=None, test_df=None, **kwargs):
    x_data = patsy_helper_nointercept(df=sample_df, formula=by)
    cell_features = prep_cell_data(df=sample_df, selected_col=by,
                                   selected_values=list(x_data.columns),
                                   fields=cell_features)
    stan_data = {'N': len(sample_df.index),
                 'G': len(sample_df.new_gene_id.unique()),
                 'S': len(sample_df.new_sample_id.unique()),
                 'C': x_data.shape[1],
                 'gene': sample_df.new_gene_id.values,
                 'sample': sample_df.new_sample_id.values,
                 'x': x_data,
                 'y': sample_df.est_counts.astype(int).values,
                 'cell_features': cell_features,
                 'M': cell_features.shape[1],
                 }
    if test_df is not None:
        x2_data = patsy_helper_nointercept(df=test_df, formula=by)
        test_data = {'N2': len(test_df.index),
                     'S2': len(test_df.new_sample_id.unique()),
                     'gene2': test_df.new_gene_id.values,
                     'sample2': test_df.new_sample_id.values,
                     'y2': test_df.est_counts.astype(int).values,
                     'x2': x2_data, ## for easy access later
                     }
        stan_data.update(test_data)
    if dict(**kwargs):
        stan_data.update(dict(**kwargs))
    return stan_data


def _list_files_in_path(path, pattern="*.stan"):
    """
    indexes a directory of stan files
    returns as dictionary containing contents of files
    """

    results = []
    for dirname, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                results.append(os.path.join(dirname, name))
    return(results)


def _find_directory(d, description=''):
    my_dir = d
    if not os.path.exists(my_dir):
        my_dir = os.path_join(_this_dir, d)
    if not os.path.exists(my_dir):
        raise ValueError('{} directory ({}) not found'.format(description, d))
    return my_dir


def _make_model_dict(model_dir, pattern="*.stan"):
    model_files = _list_files_in_path(path=model_dir, pattern=pattern)
    res = dict()
    [res.update({ntpath.basename(model_file): model_file}) for model_file in model_files]
    return res


def get_model_file(model_name, model_dir='models', pattern="*.stan"):
    clean_model_dir = _find_directory(d=model_dir, description='model')
    model_files = _make_model_dict(clean_model_dir, pattern=pattern)
    if model_name in model_files.keys():
        model_file = model_files[model_name]
    else:
        matching_files = [mfile for (mname, mfile) in model_files.items()
                      if re.match(string=mname, pattern='{}\w'.format(model_name))]
        if len(matching_files)==1:
            model_file = matching_files[0]
        elif len(matching_files)>1:
            logger.warning('Multiple files match given string. Selecting the first')
            model_file = matching_files[0]
        else:
            logger.warning('No files match given string.')
            logger.debug('Files searched include: {}'.format('\n'.join(model_files.keys())))
            model_file = None
    if not model_file:
        raise ValueError('Model could not be identified: {}'.format(model_name))
    return model_file

def get_top_genes(model_fit, sample_df, colnames, sort_by, n_genes=10):
    theta_ldf = prep_theta_summary(model_fit, sample_df=sample_df, colnames=colnames, sort_by=sort_by)
    top_genes = theta_ldf.loc[theta_ldf['mean_abs_diff_rank_{}'.format(sort_by)] <= n_genes,:] \
                .drop_duplicates(subset='new_gene_cat')['new_gene_cat'].values
    return top_genes

def plot_posterior_predictive_checks(model_fit, plot_genes, sample_df, yrep_df=None, n_genes=None):
    if not n_genes:
        n_genes = len(plot_genes)
    plot_genes = plot_genes[:n_genes]
    if yrep_df is None:
        yrep_df = prep_yrep_summary(model_fit=model_fit, sample_df=sample_df, filter_genes=plot_genes)
    # plot estimates & observed values for top 3 genes, by Subset
    with sns.plotting_context('talk'):
        if n_genes > 1:
            f, axarr = plt.subplots(3, n_genes, sharey=False, sharex=True)
        else:
            f, axarr = plt.subplots(3, sharey=False, sharex=True)
        cell=0
        for cell_type in ['CD4','CD8','B']:
            gene=0
            for gene_name in plot_genes:
                if n_genes > 1:
                    this_ax = axarr[cell, gene]
                else:
                    this_ax = axarr[cell]
                yrep_data = yrep_df.query('gene_cat == "{}" and cell_type == "{}"'.format(gene_name, cell_type))
                obs_data = sample_df.query('gene_cat == "{}" and cell_type == "{}"'.format(gene_name, cell_type))
                this_ax.grid(False)
                g = sns.boxplot(data=yrep_data,
                                y='SubSet',
                                x='pp_est_counts',
                                ax=this_ax,
                                fliersize=0,
                                linewidth=0.2)
                sns.swarmplot(data=obs_data,
                               y='SubSet',
                               ax=this_ax,
                               x='est_counts',
                               color='black')
                this_ax.set_xlabel('')
                gene = gene+1
            cell = cell+1

    gene=0
    for gene_name in plot_genes:
        if n_genes > 1:
            header_rows = axarr[0, gene]
            footer_rows = axarr[2, gene]
        else:
            header_rows = axarr[0]
            footer_rows = axarr[2]
        header_rows.set_title(gene_name)
        plt.setp(footer_rows.get_xticklabels(), rotation='vertical')
        footer_rows.set_xlabel('estimated counts')

        if (gene > 0):
            for cell in np.arange(3):
                plt.setp(axarr[cell, gene].get_yticklabels(), visible=False)
        gene = gene+1

    

def run_model(model_name, sample_n=500, by='cell_type', cell_features=None,
              model_dir='models', iter=500, chains=4, **kwargs):
    model_file = get_model_file(model_name=model_name, model_dir=model_dir)
    sample_df = cached(prep_sample_df, sample_n=sample_n)
    stan_data = prep_stan_data(sample_df, by=by, cell_features=cell_features)
    fit = cached_stan_fit(file=model_file, data=stan_data, iter=iter, chains=chains, model_name=model_name, **kwargs)
    return dict(fit=fit, stan_data=stan_data, sample_df=sample_df, model_file=model_file, model_name=model_name)
