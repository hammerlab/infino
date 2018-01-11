import numpy as np
import pandas as pd
import argparse

def _chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    https://stackoverflow.com/a/312464/130164
    """
    for i in range(0, len(l), n):
        yield list(l[i:i + n]) # convert from ndarray to list if ndarray


def chunker(input_name, output_fname_template, pagesize, min_group_size=1):
    df = pd.read_csv(input_name, sep='\t', index_col=0)
    samples = df.columns.values

    # make chunks
    chunks = list(_chunks(samples, pagesize))

    # last chunk might be too small (e.g. remainder of 1)
    # if any(len(c) < min_group_size for c in chunks)
    if len(chunks) > 1 and len(chunks[-1]) < min_group_size:
        # stick those samples in other chunks
        for sample in chunks[-1]:
            # destination_chunk = np.random.choice(chunks[:-1], 1) # argument to choice must be one-dimensional, so can't just pass chunks[:-1]
            destination_chunk_ix = np.random.randint(0, len(chunks) - 1) # rand int between 0 (inclusive) and len(chunks)-1 (exclusive)
            chunks[destination_chunk_ix].append(sample)
        chunks = chunks[:-1]  # remove last chunk

    # confirm our manipulation didn't destroy anything
    assert np.sum([len(c) for c in chunks]) == len(samples)

    # construct dataframes
    chunk_dfs = [df[c] for c in chunks]

    # file names
    output_fnames = [output_fname_template.format(
        x=idx+1, y=len(chunk_dfs)) for idx, chunk_df in enumerate(chunk_dfs)]

    # output
    for chunk_df, output_fname in zip(chunk_dfs, output_fnames):
        chunk_df.to_csv(output_fname, sep='\t')

    return chunks, output_fnames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help="path to test data (tab delimited genes x samples)")
    parser.add_argument('--output_file', required=False,
                        help='output file prefix (optional)')
    parser.add_argument('--pagesize', required=True, type=int,
                        default=10, help='page size (default 10)')
    parser.add_argument('--min_chunk_size', default=3, type=int,
                        help='minimum number of samples in a chunk (default 3), achieved via rebalancing (set to 1 to disable)')

    args = parser.parse_args()

    output_fname_template = (
        args.data if not args.output_file else args.output_file) + '.chunk-{x}-of-{y}.tsv'

    chunks, output_fnames = chunker(
        args.data, output_fname_template, args.pagesize, args.min_chunk_size)

    print("Output:")
    for chunk, output_fname in zip(chunks, output_fnames):
        if len(chunk) == 1:
            print(output_fname, " -- 1 sample (%s)" % chunk[0])
        else:
            print(output_fname, " -- %d samples (%s, ..., %s)" %
                  (len(chunk), chunk[0], chunk[-1]))

if __name__ == '__main__':
    main()