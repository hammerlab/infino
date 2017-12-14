
# fix matplotlib imports in the below packages
import matplotlib
matplotlib.use('Agg')

from pystan.misc import stan_rdump
from pystan.constants import MAX_UINT
import random
import numpy as np
import pandas as pd
import subprocess
import argparse
import models

# import pdb; pdb.set_trace()


def make_training_stan_dict(df, map_sample_to_subset):
    """
    input: genes x samples dataframe; a mapping from sample name to subset
    returns: training part of stan dictionary
    """
    # columns that models.prep_stan_data expects: new_sample_id; new_gene_id (not gene_name); est_counts; SubSet
    # TODO: how to get cell features in there? doesn't seem like we have so far
    tmp_df = df.reset_index() # this takes gene_name into first columnn
    training_df = pd.melt(
        tmp_df, 
        id_vars=tmp_df.columns[0],
        var_name='sample_name', # might not be an int
        value_name='est_counts' # this is what models.prep_stan_data expects; not necessarily raw counts
    ).rename(columns={'index': 'gene_name'})

    # map sample names to subsets
    training_df['SubSet'] = training_df['sample_name'].apply(lambda x: map_sample_to_subset.loc[x].values[0])

    # create gene and sample IDs
    training_df['new_sample_cat'] = training_df['sample_name'].astype('category')
    training_df['new_sample_id'] = training_df['new_sample_cat'].cat.codes+1

    training_df['new_gene_cat'] = training_df['gene_name'].astype('category')
    training_df['new_gene_id'] = training_df['new_gene_cat'].cat.codes+1

    # extract the map of gene name to gene ID
    map_gene_name_to_id = training_df[['gene_name', 'new_gene_id']].drop_duplicates().set_index('gene_name')
    assert map_gene_name_to_id.shape[0] == training_df.gene_name.nunique()

    stan_data_dict = models.prep_stan_data(
        sample_df=training_df,
        test_df=None,
        by='SubSet',
        #y=models['est_counts'].astype(int).values, # in case want to replace with another column -- do it here
    )
    assert all(type(t) == np.int64 for t in stan_data_dict['y']) # required, because negative binomial distribution..
    return stan_data_dict, map_gene_name_to_id



def make_test_stan_dict(df, map_gene_name_to_id):
    """
    takes a genes x samples dataframe, and converts it into test part of stan dictionary
    """

    tmp_df = df.reset_index() # this takes gene_name into first columnn
    testing_df = pd.melt(
        tmp_df, 
        id_vars=tmp_df.columns[0],
        var_name='sample_name', # might not be an int
        value_name='value' 
    ).rename(columns={'index': 'gene_name'})

    # create sample IDs
    testing_df['new_sample_cat'] = testing_df['sample_name'].astype('category')
    testing_df['new_sample_id'] = testing_df['new_sample_cat'].cat.codes+1

    # extract the map of sample IDs
    map_sample_name_to_id = testing_df[['sample_name', 'new_sample_id']].drop_duplicates()
    assert map_sample_name_to_id.shape[0] == testing_df.sample_name.nunique()

    # map gene names to IDs to match training DF
    testing_df['new_gene_id'] = testing_df['gene_name'].apply(lambda x: map_gene_name_to_id.loc[x].values[0])
    assert not testing_df['new_gene_id'].isnull().any()

    stan_data_dict = {
             'N2': len(testing_df.index),
             'S2': len(testing_df.new_sample_id.unique()),
             'gene2': testing_df.new_gene_id.values,
             'sample2': testing_df.new_sample_id.values,
             'y2': testing_df['value'].astype(int).values, # crucial to set as int (for negative binomial)
             }

    assert all(type(t) == np.int64 for t in stan_data_dict['y2']) # required, because negative binomial distribution..
    return stan_data_dict, map_sample_name_to_id


# based on pystan documentation
generate_seed = lambda : random.randint(0, MAX_UINT)


def generate_chain_command(**kwargs):
    sample_log_fname = "{experiment_name}.samples.{chain_id}.csv".format(**kwargs)
    stdout_fname = "{experiment_name}.stdout.{chain_id}.txt".format(**kwargs)
    seed_fname = "{experiment_name}.seed.{chain_id}.txt".format(**kwargs) # output seed for reproducibility
    kwargs['output_fname'] = sample_log_fname
    command_template = """
        echo {modelexe} method=sample num_samples=1000 num_warmup=1000 save_warmup=0 thin=1 \\
        random seed={seed} \\
        id={chain_id} data file={data_fname} \\
        output file={output_fname} refresh=25
        """.format(**kwargs)
    return command_template, sample_log_fname, stdout_fname, seed_fname


def announce_progress(progress_str, log_file_handler, chain_id):
    """
    print progress to this process's stdout and to [log_file_handler]
    prefix with [chain_id]
    """
    log_text = '[Chain %d] %s' % (chain_id, progress_str)
    print(log_text)
    log_file_handler.write(log_text + '\n')

if __name__ == '__main__':

    # load command line arguments

    parser = argparse.ArgumentParser()


    parser.add_argument('--train_samples', required=True, help='training matrix filename')
    parser.add_argument('--train_celltypes', required=True, help='map from training samples to subsets')
    parser.add_argument('--test_samples', required=True, help='test matrix filename (to be deconvolved)')
    parser.add_argument('--n_chains', default=4, help='number of MCMC chains')
    parser.add_argument('--output_name', required=True, help='prefix for output files (include chunk number here!)')
    parser.add_argument('--model_executable', required=True, help='compiled stan model')

    args = parser.parse_args()


    # load in train and test files
    train_df = pd.read_csv(args.train_samples, index_col=0, sep='\t')
    train_df.columns = train_df.columns.astype(str) # so that compatible with train_sample_map's sample_name
    train_sample_map = pd.read_csv(args.train_celltypes, sep='\t')
    assert train_sample_map['sample_name'].nunique() == train_sample_map['sample_name'].count()
    train_sample_map['sample_name'] = train_sample_map['sample_name'].astype(str)
    train_sample_map.set_index('sample_name', inplace=True)
    test_df = pd.read_csv(args.test_samples, index_col=0, sep='\t')

    # confirm they have exactly the same genes
    assert len(set(train_df.index.values).symmetric_difference(set(test_df.index.values))) == 0

    # create stan data dicts, recoding sample IDs
    train_dict, map_gene_name_to_id = make_training_stan_dict(train_df, train_sample_map)
    test_dict, map_sample_name_to_id = make_test_stan_dict(test_df, map_gene_name_to_id)

    # output sample ID test chunk map
    map_sample_name_to_id.to_csv(args.output_name + '.sample_name_to_id.tsv', sep='\t')
     
    # combine them
    train_dict.update(test_dict)

    # convert to Rdump
    for key in train_dict.keys():
        # stan_rdump requires ndarray/matrix rather than pandas dataframes
        # affects keys including: 'x', 'cell_features'
        if type(train_dict[key]) == pd.DataFrame:
            train_dict[key] = train_dict[key].as_matrix()

    data_fname = args.output_name + '.standata.Rdump'
    stan_rdump(train_dict, data_fname)


    # generate commands to run and files to write to
    chains = []
    for i in range(args.n_chains):
        seed = generate_seed()
        command, sample_log_fname, stdout_fname, seed_fname = generate_chain_command(
            seed=seed,
            experiment_name=args.output_name,
            chain_id=i+1,
            modelexe = args.model_executable,
            sample_log_fname = args.output_name + '.samples.log',
            data_fname = data_fname
        )
        chains.append({
                'chain_id': i+1,
                'command': command, 
                'sample_log': sample_log_fname,
                'stdout_log': stdout_fname,
                'seed': seed,
                'seed_fname': seed_fname
            }
        )

    # launch processes
    for chain in chains:
        print("Launching chain: ", chain['chain_id'])
        proc = subprocess.Popen(
            chain['command'],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        chain['proc'] = proc
        chain['stdout_file_handler'] = open(chain['stdout_log'], 'w')
        # output seed for reproducibility
        with open(chain['seed_fname'], 'w') as seed_f:
            seed_f.write(chain['seed'])

    # monitor progress, write to stdout and console log file
    while any(chain['proc'].returncode is None for chain in chains):
        for chain in chains:
            try:
                announce_progress(str(chain['proc'].communicate()), chain['stdout_file_handler'], chain['chain_id'])
            except Exception as e:
                print("Error getting stdout/err from chain %d:" % chain['chain_id'], e, "-- continuing.")

    # after all chains complete, continue
    for chain in chains:
        # flush any announcements
        try:
            announce_progress(str(chain['proc'].communicate()), chain['stdout_file_handler'], chain['chain_id'])
        except Exception as e:
            # i think this actually is supposed to fail always?
            pass
        
        # get cmdstan exit code
        announce_progress("Execution completed, return code: %d" % chain['proc'].returncode, chain['stdout_file_handler'], chain['chain_id'])
        
        # close out the file handler
        chain['stdout_file_handler'].close()

    ## run stansummary
    # sample log filenames are available in chains object.
    # since we got here, all chains must have written proper logs -- so we will avoid the bug of stansummary erroring out in the case of broken chains
    # TODO

    # TODO: print timing details from end of chain sampling log files?



