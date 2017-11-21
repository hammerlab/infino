# *Infino*: a Bayesian hierarchical model improves estimates of immune infiltration into tumor microenvironment

[Preprint (TODO)](#)

## Install

We distribute a batteries-included environment for running infino via a Docker image. Please see the instructions at: https://hub.docker.com/r/hammerlab/infino-docker/

This process involves cloning https://github.com/hammerlab/immune-infiltrate-explorations, where a lot of source code currently resides.

## Use

We are actively refining the way we distribute infino's source code and training data. For your evaluation while we prepare a cleaner environment, we outline the steps involved in running infino:

1. Download training data. We are working to release our processed version of [the training data used in the paper](https://www.nature.com/articles/sdata201551). The processing was done by running [these Kubernetes jobs](https://github.com/hammerlab/infiltrate-rnaseq-pipeline/). Please see the runbook in that repository for our exact notes from downloading the training data from the original source, preprocessing that data, and then running Kallisto.

    You can prepare more training data with [a protocol such as this, which includes a video demonstration](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144656/).

2. Extract raw counts or TPM data for samples of unknown composition that you'd like to deconvolve with infino.

3. Combine the training and test data, then run ComBat to perform batch correction. Barring any additional information about the provenance of your test data (and any additions you make to the training data), you can use at least two batches: one for the provided training data, and one for your separate test data.

    An example of batch correction is included in the two jupyter notebooks in this repo labeled "TCGA-KIRC". They also demonstrate how to transform the batch-corrected data into the proper input format for Stan. 

4. Run [the infino model](https://github.com/hammerlab/immune-infiltrate-explorations/blob/master/model-single-origin-samples/models/model6.2_negbinom_matrix_correlation_features_oos_optim.stan), e.g.:

    ```bash
    # tcgakirc_cut1_tpm_raw: tcgakirc.singleorigin.stan_data.tpm_corrected.pkl, tcgakirc.standict.cut1.tpm.corrected.pkl

    mkdir -p logs/tcgakirc_cut1_tpm_corrected && \
    chmod -R 777 logs/tcgakirc_cut1_tpm_corrected && \
    docker run -it -d --name tcgakirc_cut1_tpm_corrected \
      -v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
      -v $(pwd):/home/jovyan/work/model-single-origin-samples/run/ \
      hammerlab/infino-docker:latest \
      bash -c "cd work/model-single-origin-samples/run/ && \
        python \
          cohort_runstan.py \
          tcgakirc.singleorigin.stan_data.tpm_corrected.pkl \
          tcgakirc.standict.cut1.tpm.corrected.pkl \
          tcgakirc_cut1_tpm_corrected 2>&1 | tee \
          logs/tcgakirc_cut1_tpm_corrected/run_stan.tcgakirc_cut1_tpm_corrected.consoleout.txt"
    ```

5. Run `stansummary`, e.g.:

    ```bash
    docker run -it -d --name stansum_tcgakirc_cut1_tpm_corrected \
    -v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
    -v $(pwd):/home/jovyan/work/model-single-origin-samples/run/ \
    hammerlab/infino-docker:latest \
    bash -c "cd work/model-single-origin-samples/run/ && stansummary \
    --csv_file=logs/tcgakirc_cut1_tpm_corrected/stansummary.tcgakirc_cut1_tpm_corrected.csv \
    logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_0.csv \
    logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_1.csv \
    logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_2.csv \
    logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_3.csv > /dev/null;"
    ```

6. Generate diagnostics and plots with `analyze_cut.py`:

    ```bash
    mkdir -p plots;
    chmod -R 777 plots;
    
    docker run -it -d --name plot_tcgakirc_cut1_tpm_corrected \
    -v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
    -v $(pwd):/home/jovyan/work/model-single-origin-samples/run/ \
    hammerlab/infino-docker:latest \
    bash -c "cd work/model-single-origin-samples/run/ && python analyze_cut.py \
    --stansummary logs/tcgakirc_cut1_tpm_corrected/stansummary.tcgakirc_cut1_tpm_corrected.csv \
    --slug tcgakirc_cut1_tpm_corrected \
    --cohort_name tcgakirc \
    --cohort_name_cib tcgakirc \
    --metric tpm \
    --processing corrected \
    --cutname cut1 \
    --trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_0.csv \
    --trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_1.csv \
    --trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_2.csv \
    --trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_3.csv \
    ;"
    ```

    You can see example output from `analyze_cut.py` in `example analysis.ipynb`, which was used to generate the standalone python script.

