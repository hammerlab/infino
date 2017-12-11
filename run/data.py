from __future__ import print_function
import numpy as np
import pandas as pd
import pyensembl
from pyensembl import cached_release
import os
from subprocess import call
import tarfile
import logging 
from cache import cached

logger = logging.getLogger(__name__)

def download(s, data_dir='./data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ## download & extract data tar file
    data_tar_file = '{}.tar.gz'.format(s)
    call(['gsutil', 'cp', 'gs://mz-hammerlab/output/{}'.format(data_tar_file), data_dir])
    data_tar = tarfile.open(os.path.join(data_dir, data_tar_file))
    data_tar.extractall(path=data_dir)
    data_tar.close()
    ## download & extract log tar file
    log_tar_file = '{}.logs.tar.gz'.format(s)
    call(['gsutil', 'cp', 'gs://mz-hammerlab/output/{}'.format(log_tar_file), data_dir])
    log_tar = tarfile.open(os.path.join(data_dir, data_tar_file))
    log_tar.extractall(path=data_dir)
    log_tar.close()
    print('downloaded', s, 'files to', data_dir)
    os.remove(os.path.join(data_dir, data_tar_file))
    os.remove(os.path.join(data_dir, log_tar_file))


def load_multiple_files(files, ensembl_release=cached_release(79)):
    """
    files is ordered list of samples. they will be assigned sample ids 1-n.
    """
    dfs = []
    for ix, f in enumerate(files):
        filename = '/data/output/%s/abundance.tsv' % f
        if not os.path.exists(filename):
            download(f)
        df = pd.read_csv(filename, sep='\t')
        df['sample_id'] = ix+1
        df['filename'] = f
        df['gene_name'] = df['target_id'].map(lambda t: ensembl_release.gene_name_of_transcript_id(t))
        df['log1p_tpm'] = np.log1p(df['tpm'])
        dfs.append(df)
    return pd.concat(dfs)


def prep_simple_summary(df):
    """
    sum counts, abundance (tpm) across genes
    """
    return df.groupby(['sample_id', 'filename', 'gene_name'])[['est_counts', 'tpm']].sum().reset_index()


## inspired by 
## http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows
def split_rows_by(df, field, suffix='', by=','):
    s = df[field].str.split(by).apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = '{}{}'.format(field, suffix)
    if suffix == '':
        del df[field]
    return df.join(s)


def prep_filename_metadata(datafiles=None):
    if datafiles is None:
        datafiles = pd.read_csv('data_filenames.tsv', sep='\t')
    df = datafiles[['filename', 'SubSet', 'Antibody']].copy()
    subsets = split_rows_by(df, 'Antibody')
    #subsets['antibody'] = subsets.Antibody.replace(to_replace='.$', value='', inplace=False, regex=True)
    #subsets['value'] = subsets.Antibody.replace(to_replace='^.+(.)$', value='\\1', inplace=False, regex=True)
    #subsets['value'] = subsets['value'].apply(lambda x: 0 if not x or x == '' else 1 if x == '+' else 2 if x == '-' else 0)
    subsets['antibody'] = subsets.Antibody
    subsets['value'] = 1
    del subsets['Antibody']
    del subsets['SubSet']
    subsets.drop_duplicates(['filename','antibody', 'value'], inplace=True)
    subsets = subsets.pivot('filename', 'antibody', 'value')
    subsets.reset_index(inplace=True)
    subsets = pd.merge(subsets, df.loc[:,['filename','SubSet']], on='filename', how='outer')
    subsets.fillna(0, inplace=True)
    return subsets


def load_by_cell_type(cell_types=None, metadata=None):
    if metadata is None:
        metadata = prep_filename_metadata()
    if cell_types:
        file_ids = []
        [file_ids.extend(list(metadata.loc[metadata.SubSet == cell_type, 'filename'].values)) for cell_type in cell_types]
    else:
        file_ids = metadata.drop_duplicates(subset='filename')['filename'].values
    simple_summary_data = prep_simple_summary(load_multiple_files(file_ids))
    simple_summary_data['log1p_tpm'] = np.log1p(simple_summary_data['tpm'])
    simple_summary_data['log1p_counts'] = np.log1p(simple_summary_data['est_counts'])
    return pd.merge(simple_summary_data, metadata, on='filename', how='left')


def gmean(x):
    return np.expm1(np.mean(np.log1p(x)))


def gstd(x):
    return np.expm1(np.std(np.log1p(x)))


def rescale_geom(x):
    return (x - gmean(x)) / gstd(x) if gstd(x)>0 else 0


def prep_annotated_data(df=None):
    if df is None:
        df = load_by_cell_type()
    df['cell_type'] = df['SubSet'].apply(lambda x: x.split('_')[0])
    df['log1p_tpm_rescaled_type'] = df \
            .groupby(['cell_type','gene_name'])['log1p_tpm'] \
            .transform(rescale_geom)
    df['log1p_tpm_rescaled_subset'] = df \
            .groupby(['SubSet','gene_name'])['log1p_tpm'] \
            .transform(rescale_geom) 
    df['log1p_tpm_rescaled'] = df \
            .groupby(['gene_name'])['log1p_tpm'] \
            .transform(rescale_geom)
    df['gene_cat'] = df['gene_name'].astype('category')
    df['gene_id'] = df['gene_cat'].cat.codes+1
    df['B_cell'] = df['cell_type'].apply(lambda x: 1 if x == 'B' else 0)
    df['T_cell'] = df['cell_type'].apply(lambda x: 1 if x != 'B' else 0)
    return df



