map_host_to_task = {
        'infino-run-1.us-east1-b.pici-hammerlab': """
docker run -it -d --name experiment_rcc-test-chunked-chunk-1-of-2-model6_4 \
-v $HOME/infino:/home/jovyan/work \
hammerlab/infino-docker:develop \
bash -c "cd work/ && execute-model \
--train_samples notebooks/out/experiment_rcc.training.expression.tsv \
--train_xdata notebooks/out/experiment_rcc.training.xdata.tsv \
--train_cellfeatures notebooks/out/singleorigin_plus_rcctils.combined.cellfeatures.tsv \
--test_samples notebooks/out/experiment_rcc.test.chunked.chunk-1-of-2.tsv \
--n_chains 4 \
--output_name out/experiment_rcc.test.chunked.chunk-1-of-2.model6_4 \
--model_executable models/model6.4_negbinom_matrix_correlation_features_oos_optim_otherbucket \
--num_samples 1000 \
--num_warmup 1000 \
2>&1 | tee out/experiment_rcc.test.chunked.chunk-1-of-2.model6_4.consoleout.txt"        
        """,
#         'infino-run-2.us-east1-b.pici-hammerlab': """
# # tcgakirc_cut1_tpm_raw: tcgakirc.singleorigin.stan_data.tpm_raw.pkl, tcgakirc.standict.cut1.tpm.raw.pkl
# mkdir -p logs/tcgakirc_cut1_tpm_raw && chmod -R 777 logs/tcgakirc_cut1_tpm_raw && docker run -it -d --name tcgakirc_cut1_tpm_raw \
# -v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
# hammerlab/infino-docker:latest \
# bash -c "cd work/model-single-origin-samples/infino-rcc/ && python \
# cohort_runstan.py tcgakirc.singleorigin.stan_data.tpm_raw.pkl tcgakirc.standict.cut1.tpm.raw.pkl tcgakirc_cut1_tpm_raw 2>&1 | tee logs/tcgakirc_cut1_tpm_raw/run_stan.tcgakirc_cut1_tpm_raw.consoleout.txt"
#         """,
#         'infino-3.us-east1-b.pici-hammerlab': """
# # tcgakirc_cut1_counts_corrected: tcgakirc.singleorigin.stan_data.counts_corrected.pkl, tcgakirc.standict.cut1.counts.corrected.pkl
# mkdir -p logs/tcgakirc_cut1_counts_corrected && chmod -R 777 logs/tcgakirc_cut1_counts_corrected && docker run -it -d --name tcgakirc_cut1_counts_corrected \
# -v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
# hammerlab/infino-docker:latest \
# bash -c "cd work/model-single-origin-samples/infino-rcc/ && python \
# cohort_runstan.py tcgakirc.singleorigin.stan_data.counts_corrected.pkl tcgakirc.standict.cut1.counts.corrected.pkl tcgakirc_cut1_counts_corrected 2>&1 | tee logs/tcgakirc_cut1_counts_corrected/run_stan.tcgakirc_cut1_counts_corrected.consoleout.txt"
#         """,
}
        
    
map_host_to_docker_name = {
        'infino-run-1.us-east1-b.pici-hammerlab': "experiment_rcc-test-chunked-chunk-1-of-2-model6_4",
        # 'infino-2.us-east1-b.pici-hammerlab': "tcgakirc_cut1_tpm_raw",
        # 'infino-3.us-east1-b.pici-hammerlab': "tcgakirc_cut1_counts_corrected",
}