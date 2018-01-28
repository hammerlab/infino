docker run -it --rm -v $(pwd):/home/jovyan/work:ro hammerlab/infino-docker:latest bash
> python execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe
# or for debug
> python -i execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe

actually run without :ro
docker run -it --rm -v $(pwd):/home/jovyan/work hammerlab/infino-docker:latest bash
> cd work
> python -i execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe --train_celltypes=private_data/map_sample_to_celltype.txt


> python chunker.py --data private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --output_file testout --pagesize 30 --min_chunk_size 10

Create executable version of `modelname.stan`: `make -C $HOME/cmdstan $(pwd)/modelname`. Creates `modelname` executable. May need to clear out manually to force make rerun: `rm modelname.hpp modelname`