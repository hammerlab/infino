
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
from .run_stansummary import stansummary
import pickle

# import pdb; pdb.set_trace()


def make_training_stan_dict(df, sample_x_matrix, cell_features):
    """
    @param df: genes x samples dataframe. must have sample names header row.
    @param sample_x_matrix: an x_data design matrix (samples x subsets) that indicates fractional presence of all subsets within each sample. 
    @param cell_features: subsets x markers dataframe (values are 0 or 1 for absense / presence of marker for a certain cell type)
    returns: training part of stan dictionary
    """

    # melt the G x S dataframe 
    tmp_df = df.reset_index() # this takes gene_name into first columnn
    training_df = pd.melt(
        tmp_df, 
        id_vars=tmp_df.columns[0],
        var_name='sample_name', # might not be an int
        value_name='est_counts' # this name is carried over from what models.prep_stan_data expected; not necessarily raw counts!!
    ).rename(columns={'index': 'gene_name'})

    # confirm that x_data has what we need (i.e. that we have xdata for every sample)
    assert df.columns.astype(str).equals(sample_x_matrix.index.astype(str))
    # ensure we are in matched order
    x_data = sample_x_matrix.copy()
    x_data.index = x_data.index.astype(str)
    x_data = x_data.loc[df.columns.astype(str)]
    
    # confirm that cell_features has what we need
    assert all(c in cell_features.index.values for c in x_data.columns)

    # filter cell features to current cell types, and order it in the same order
    cell_features_df = cell_features.loc[x_data.columns].copy()
    # fill blanks
    cell_features_df.fillna(0, inplace=True)
    # remove any markers that are not present in current cell types -- i.e. any columns that have all 0s
    cell_features_df = cell_features_df.loc[:, (cell_features_df != 0).any(axis=0)]

    # create gene and sample IDs
    training_df['new_sample_cat'] = training_df['sample_name'].astype('category')
    training_df['new_sample_id'] = training_df['new_sample_cat'].cat.codes+1

    training_df['new_gene_cat'] = training_df['gene_name'].astype('category')
    training_df['new_gene_id'] = training_df['new_gene_cat'].cat.codes+1

    # extract the map of gene name to gene ID
    map_gene_name_to_id = training_df[['gene_name', 'new_gene_id']].drop_duplicates().set_index('gene_name')
    assert map_gene_name_to_id.shape[0] == training_df.gene_name.nunique()

    x_data_reshaped = x_data.loc[training_df.sample_name.values] # it's really N x C -- and the N has to be arranged following the 'sample' list 

    stan_data_dict = {
        'N': len(training_df.index), # total number of observations (GxS)
        'G': len(training_df.new_gene_id.unique()), # number of genes
        'S': len(training_df.new_sample_id.unique()), # number of samples
        'C': x_data.shape[1], # number of subsets
        'gene': training_df.new_gene_id.values, # gene ID value for each of the N observations
        'sample': training_df.new_sample_id.values, # sample ID value for each of the N observations
        #'x': x_data, # S x C matrix 
        'x': x_data_reshaped, # N x C matrix
        'y': training_df.est_counts.astype(int).values, # very important to make it int -- because negative binomial distribution
        'cell_features': cell_features, # C x M matrix
        'M': cell_features.shape[1],
    }


    assert len(stan_data_dict['y']) == stan_data_dict['N'] # int<lower=0> y[N]; // count/tpm for each obs
    # y must correspond: sample_y[sample[n], gene[n]] = y[n];
    # this is guaranteed in our construction because we take G, S, y directly from `training_df` without any reordering
    assert all(stan_data_dict['y'] >= 0), 'Values must be non-negative'

    # confirm vector<lower=0, upper=1>[C] x[N];
    assert stan_data_dict['x'].shape == (stan_data_dict['N'], stan_data_dict['C'])
    # next two assertions removed because they assume that our sample IDs go 1,1,1,1,1,1,12,12,12,12,12,12,.. as opposed to 1,12,3,4,5,1,12,3,4,5..
    #assert np.array_equal(stan_data_dict['x'][0,:], stan_data_dict['x'][1,:])
    #assert np.array_equal(stan_data_dict['x'][stan_data_dict['G'],:], stan_data_dict['x'][stan_data_dict['G']+1,:])
    # choose two rows that should belong to the same sample. make sure they're identical
    assert np.array_equal(
        stan_data_dict['x'].iloc[np.where(stan_data_dict['sample'] == stan_data_dict['sample'].max())[0][0]].values,
        stan_data_dict['x'].iloc[np.where(stan_data_dict['sample'] == stan_data_dict['sample'].max())[0][-1]].values
    )

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
    """
    generates the cmdstan command
    """
    sample_log_fname = "{experiment_name}.samples.{chain_id}.csv".format(**kwargs)
    stdout_fname = "{experiment_name}.stdout.{chain_id}.txt".format(**kwargs)
    seed_fname = "{experiment_name}.seed.{chain_id}.txt".format(**kwargs) # output seed for reproducibility

    # method is optional, but if it's VI then we will switch to variational inference
    method_str = "method=sample num_samples={num_samples} num_warmup={num_warmup} save_warmup=0 thin=1".format(**kwargs)
    method = kwargs.get('method', '')
    if method.lower() == 'vi':
        # method=variational iter=10000 output_samples=1000
        method_str = "method=variational iter={adjusted_iter} num_warmup={adjusted_output_samples}".format(
            adjusted_iter = kwargs['num_samples'] + kwargs['num_warmup'],
            adjusted_output_samples = kwargs['num_samples']
        )
    
    kwargs['method_str'] = method_str 
    kwargs['output_fname'] = sample_log_fname
    kwargs['echo'] = 'echo' if kwargs['dry_run'] else ''
    """
    *very deliberately* not saving out warmups.
    We need to be super careful to ensure we absolutely drop warmups when necessary and don't drop any real samples ever.
    Since this is risky, we set cmdstan to not output warmups.
    Thus that there is no need for an end user to remember that they must calculate and pass n_warmups=n_total/2 to analyze-infino to get the right results.
    """
    command_template = """
        {echo} {modelexe} {method_str} \\
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
    #log_text = '[Chain %d] %s' % (chain_id, progress_str)

    # progress_str may have multiple lines in it
    log_text = '\n'.join(['[Chain %d] %s' % (chain_id, progress_str_part) for progress_str_part in progress_str.split('\n')])

    print(log_text)
    log_file_handler.write(log_text + '\n')

def main():

    # load command line arguments

    parser = argparse.ArgumentParser()


    parser.add_argument('--train_samples', required=True, help='training matrix filename')
    parser.add_argument('--train_xdata', required=True, help='map from training samples to subsets (xdata design matrix')
    parser.add_argument('--train_cellfeatures', required=True, help='cell features matrix')
    parser.add_argument('--test_samples', required=True, help='test matrix filename (to be deconvolved)')
    parser.add_argument('--n_chains', default=4, type=int, help='number of MCMC chains')
    parser.add_argument('--output_name', required=True, help='prefix for output files (include chunk number here!)')
    parser.add_argument('--model_executable', required=True, help='compiled stan model')
    parser.add_argument('--dry_run', action='store_true', default=False, help="don't run expensive stan fit commands, but do everything else")
    parser.add_argument('--num_samples', default=1000, type=int, help='number of saved samples')
    parser.add_argument('--num_warmup', default=1000, type=int, help='number of warmup samples (NOT saved)')
    parser.add_argument('--vi', dest='method', action='store_const', const='vi', default='nuts',
        help="run VI instead of NUTS (recommendation: 1000 samples, 9000 warmup")

    args = parser.parse_args()


    ## load in train and test files
    # train_df: Genes x Samples
    train_df = pd.read_csv(args.train_samples, index_col=0, sep='\t')
    train_df.columns = train_df.columns.astype(str) # so that compatible with train_sample_map's sample_name

    # xdata: Samples x Subsets
    xdata = pd.read_csv(args.train_xdata, sep='\t')

    # cell features: Subsets x Markers
    cell_features = pd.read_csv(args.train_cellfeatures, sep='\t', index_col=0)

    test_df = pd.read_csv(args.test_samples, index_col=0, sep='\t')

    # confirm that train and test expression have exactly the same genes
    assert len(set(train_df.index.values).symmetric_difference(set(test_df.index.values))) == 0, "Train and test MUST have exactly the same genes!"

    # create stan data dicts, recoding sample IDs
    train_dict, map_gene_name_to_id = make_training_stan_dict(train_df, xdata, cell_features)
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
            data_fname = data_fname,
            dry_run = args.dry_run,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            method=args.method
        )
        chains.append({
                'chain_id': i+1,
                'command': command, 
                'sample_log': sample_log_fname,
                'stdout_log': stdout_fname,
                'seed': seed,
                'seed_fname': seed_fname,
                'num_samples': args.num_samples,
                'num_warmup': args.num_warmup,
                'method': args.method
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
            seed_f.write(str(chain['seed']))

    # monitor progress, write to stdout and console log file
    while any(chain['proc'].returncode is None for chain in chains):
        for chain in chains:
            try:
                announce_progress(str(chain['proc'].communicate()[0].strip()), chain['stdout_file_handler'], chain['chain_id'])
            except Exception as e:
                print("Error getting stdout/err from chain %d:" % chain['chain_id'], e, "-- continuing.")

    # after all chains complete, continue
    for chain in chains:
        # flush any announcements
        try:
            announce_progress(str(chain['proc'].communicate()[0].strip()), chain['stdout_file_handler'], chain['chain_id'])
        except Exception as e:
            # i think this actually is supposed to fail always?
            pass
        
        # get cmdstan exit code
        chain['exit_code'] = chain['proc'].returncode
        announce_progress("Execution completed, return code: %d" % chain['exit_code'], chain['stdout_file_handler'], chain['chain_id'])

        # close out the file handler
        chain['stdout_file_handler'].close()

    ## run stansummary
    # sample log filenames are available in chains object.
    # since we got here, all chains must have written proper logs -- so we will avoid the bug of stansummary erroring out in the case of broken chains
    stansummary(
        output_fname="{experiment_name}.stansummary.csv".format(experiment_name=args.output_name),
        input_names=[chain['sample_log'] for chain in chains],
        dry_run=args.dry_run,
        verbose=True # print return code even if not error
    )

    # TODO: print timing details from end of chain sampling log files?

    # pickle out the chain metadata just in case.
    for c in chains:
        # remove unpickleables
        del c['proc']
        del c['stdout_file_handler']
    pickle.dump({'chains': chains, 'cliargs': args}, open('%s.chain_metadata.pkl' % args.output_name, 'wb'))

    print("Run complete. Chain metadata pickled out.")


if __name__ == '__main__':
    main()

