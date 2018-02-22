map_host_to_task = {
        'infino-run-1.us-east1-b.pici-hammerlab': """
docker run -it -d --name experiment_rcc-test-chunked-chunk-1-of-2-model6_4 \
-v $HOME/infino:/home/jovyan/work \
hammerlab/infino-docker:develop \
bash -c "cd work/ && execute-model \
--train_samples deploy/data/experiment_rcc.training.expression.tsv \
--train_xdata deploy/data/experiment_rcc.training.xdata.tsv \
--train_cellfeatures deploy/data/singleorigin_plus_rcctils.combined.cellfeatures.tsv \
--test_samples deploy/data/experiment_rcc.test.chunked.chunk-1-of-2.tsv \
--n_chains 4 \
--output_name out/experiment_rcc.test.chunked.chunk-1-of-2.model6_4 \
--model_executable models/model6.4_negbinom_matrix_correlation_features_oos_optim_otherbucket \
--num_samples 1000 \
--num_warmup 1000 \
2>&1 | tee out/experiment_rcc.test.chunked.chunk-1-of-2.model6_4.consoleout.txt"        
        """,
        'infino-run-2.us-east1-b.pici-hammerlab': """
docker run -it -d --name experiment_rcc-test-chunked-chunk-2-of-2-model6_4 \
-v $HOME/infino:/home/jovyan/work \
hammerlab/infino-docker:develop \
bash -c "cd work/ && execute-model \
--train_samples deploy/data/experiment_rcc.training.expression.tsv \
--train_xdata deploy/data/experiment_rcc.training.xdata.tsv \
--train_cellfeatures deploy/data/singleorigin_plus_rcctils.combined.cellfeatures.tsv \
--test_samples deploy/data/experiment_rcc.test.chunked.chunk-2-of-2.tsv \
--n_chains 4 \
--output_name out/experiment_rcc.test.chunked.chunk-2-of-2.model6_4 \
--model_executable models/model6.4_negbinom_matrix_correlation_features_oos_optim_otherbucket \
--num_samples 1000 \
--num_warmup 1000 \
2>&1 | tee out/experiment_rcc.test.chunked.chunk-2-of-2.model6_4.consoleout.txt"        
        """,
        'infino-run-3.us-east1-b.pici-hammerlab': """
docker run -it -d --name experiment_bladder-test-chunked-chunk-1-of-1-model6_4 \
-v $HOME/infino:/home/jovyan/work \
hammerlab/infino-docker:develop \
bash -c "cd work/ && execute-model \
--train_samples deploy/data/experiment_bladder.training.expression.tsv \
--train_xdata deploy/data/experiment_bladder.training.xdata.tsv \
--train_cellfeatures deploy/data/singleorigin_plus_rcctils.combined.cellfeatures.tsv \
--test_samples deploy/data/experiment_bladder.test.chunked.chunk-1-of-1.tsv \
--n_chains 4 \
--output_name out/experiment_bladder.test.chunked.chunk-1-of-1.model6_4 \
--model_executable models/model6.4_negbinom_matrix_correlation_features_oos_optim_otherbucket \
--num_samples 1000 \
--num_warmup 1000 \
2>&1 | tee out/experiment_bladder.test.chunked.chunk-1-of-1.model6_4.consoleout.txt"        
        """,
}
        
    
map_host_to_docker_name = {
        'infino-run-1.us-east1-b.pici-hammerlab': "experiment_rcc-test-chunked-chunk-1-of-2-model6_4",
        'infino-run-2.us-east1-b.pici-hammerlab': "experiment_rcc-test-chunked-chunk-2-of-2-model6_4",
        'infino-run-3.us-east1-b.pici-hammerlab': "experiment_bladder-test-chunked-chunk-1-of-1-model6_4",
}