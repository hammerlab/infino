docker run -it --rm -v $(pwd):/home/jovyan/work:ro hammerlab/infino-docker:latest bash
> python execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe
# or for debug
> python -i execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe

actually run without :ro
docker run -it --rm -v $(pwd):/home/jovyan/work hammerlab/infino-docker:latest bash
> cd work
> python -i execute_model.py --train_samples=private_data/cohort_tcgakirc.cibersort.input.reference.datatype_tpm_corrected.txt --test_samples=private_data/cohort_tcgakirc.cibersort.test.tpm_corrected.tsv --output testout --model_executable testexe --train_celltypes=private_data/map_sample_to_celltype.txt