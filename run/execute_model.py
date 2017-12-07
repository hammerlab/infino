import pystan
import subprocess
import argparse
import models


def make_training_stan_dict(df):
    """
    takes a genes x samples dataframe, and converts it into training part of stan dictionary
    """
    # columns that models.prep_stan_data expects: new_sample_id; new_gene_id (not gene_name); est_counts
    # TODO: how to get cell features in there? doesn't seem like we have so far
    training_df = pd.melt(
        df.reset_index(), # this takes gene_name into a column named "index" (TODO: is it always named that? or do we have to use df.index.name or similar)
        id_vars='index',
        var_name='sample_name', # might not be an int
        value_name='est_counts' # this is what models.prep_stan_data expects; not necessarily raw counts
    ).rename(columns={'index': 'gene_name'})

    # create gene and sample IDs
    training_df['new_sample_cat'] = training_df['sample_name'].astype('category')
    training_df['new_sample_id'] = training_df['new_sample_cat'].cat.codes+1

    training_df['new_gene_cat'] = training_df['gene_name'].astype('category')
    training_df['new_gene_id'] = training_df['new_gene_cat'].cat.codes+1

    # extract the map of gene name to gene ID
    map_gene_name_to_id = training_df[['gene_name', 'new_gene_id']].drop_duplicates()
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
    
    testing_df = pd.melt(
        df.reset_index(), # this takes gene_name into a column named "index" (TODO: is it always named that? or do we have to use df.index.name or similar)
        id_vars='index',
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
    testing_df['new_gene_id'] = testing_df['gene_name'].apply(lambda x: map_gene_name_to_id.get(x, None))
    assert not testing_df['new_gene_id'].isnull().any()

    stan_data_dict = {
             'N2': len(testing_df.index),
             'S2': len(testing_df.new_sample_id.unique()),
             'gene2': testing_df.new_gene_id.values,
             'sample2': testing_df.new_sample_id.values,
             'y2': testing_df['value'].astype(int).values, # crucial to set as int (for negative binomial)
             'colname_test': relevant_col_name
             }

    assert all(type(t) == np.int64 for t in stan_data_dict['y2']) # required, because negative binomial distribution..
    return stan_data_dict, map_sample_name_to_id


# based on pystan documentation
generate_seed = lambda : random.randint(0, pystan.constants.MAX_UINT)


def generate_chain_command(**kwargs):
    sample_log_fname = "{experiment_name}.samples.{chain_id}.csv".format(**kwargs)
    stdout_fname = "{experiment_name}.stdout.{chain_id}.txt".format(**kwargs)
    command_template = """
        {modelexe} method=sample num_samples=1000 num_warmup=1000 save_warmup=0 thin=1 \\
        random seed={seed} \\
        id={chain_id} data file={} \\
        output file={sample_log_fname} refresh=25
        """.format(**kwargs).format(output_fname=sample_log_fname)
    return command_template, sample_log_fname, stdout_fname


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


    parser.add_argument('--train_samples', required=True)
    parser.add_argument('--test_samples', required=True)
    parser.add_argument('--n_chains', default=4)
    parser.add_argument('--output_name', required=True)
    parser.add_argument('--model_executable', required=True)

    args = parser.parse_args()


    # load in train and test files
    train_df = pd.read_csv(args['train_samples'])
    test_df = pd.read_csv(args['test_samples'])

    # TODO: confirm they have exactly the same genes

    # create stan data dicts, recoding sample IDs
    train_dict, map_gene_name_to_id = make_training_stan_dict(train_df)
    test_dict, map_sample_name_to_id = make_test_stan_dict(test_df, map_gene_name_to_id)

    # TODO: output sample ID test chunk map
     
    # combine them
    train_dict.update(test_dict)

    # TODO convert to Rdump : pystan.misc.stan_rdump(data, filename)


    # generate commands to run and files to write to
    chains = []
    for i in range(n_chains):
        command, sample_log_fname, stdout_fname = generate_chain_command( # TODO
            seed=generate_seed()
            )
        chains.append({
                'chain_id': i+1,
                'command': command, 
                'sample_log': sample_log_fname,
                'stdout_log': stdout_fname
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

    # TODO: run stansummary: sample log filenames are available in chains object.

    # TODO: print timing details from end of chain sampling log files?



