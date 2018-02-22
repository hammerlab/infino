# Experiments to run

* RCC chunk 1 -- model 6.4 and 6.5
* RCC chunk 2 -- model 6.4 and 6.5
* Bladder chunk 1 -- model 6.4 and 6.5

TPM as metric w/ batch correction for all.

For now, we are only scheduling model 6.4.

# Provision machines

Need three machines:

```bash
gcloud beta compute --project "pici-hammerlab" instances create "infino-run-1" \
--zone "us-east1-b" --machine-type "n1-highmem-4" --subnet "default" --no-restart-on-failure \
--maintenance-policy "MIGRATE" \
--service-account "infino@pici-hammerlab.iam.gserviceaccount.com" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--min-cpu-platform "Automatic" \
--image "ubuntu-1604-xenial-v20171026a" --image-project "ubuntu-os-cloud" \
--boot-disk-size "60" --boot-disk-type "pd-standard" --boot-disk-device-name "infino-run-1";

gcloud beta compute --project "pici-hammerlab" instances create "infino-run-2" \
--zone "us-east1-b" --machine-type "n1-highmem-4" --subnet "default" --no-restart-on-failure \
--maintenance-policy "MIGRATE" \
--service-account "infino@pici-hammerlab.iam.gserviceaccount.com" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--min-cpu-platform "Automatic" \
--image "ubuntu-1604-xenial-v20171026a" --image-project "ubuntu-os-cloud" \
--boot-disk-size "60" --boot-disk-type "pd-standard" --boot-disk-device-name "infino-run-2";

gcloud beta compute --project "pici-hammerlab" instances create "infino-run-3" \
--zone "us-east1-b" --machine-type "n1-highmem-4" --subnet "default" --no-restart-on-failure \
--maintenance-policy "MIGRATE" \
--service-account "infino@pici-hammerlab.iam.gserviceaccount.com" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--min-cpu-platform "Automatic" \
--image "ubuntu-1604-xenial-v20171026a" --image-project "ubuntu-os-cloud" \
--boot-disk-size "60" --boot-disk-type "pd-standard" --boot-disk-device-name "infino-run-3";

```


If you make a mistake and need to reset fully:

```
# think before running these!
gcloud compute instances stop infino-run-1 infino-run-2 infino-run-3; 
gcloud compute instances delete --delete-disks=all --quiet infino-run-1 infino-run-2 infino-run-3;
```

# Set up SSH connections

Then: `gcloud compute config-ssh` (makes edits to `~/.ssh/config`). After this, you can access any node directly with `ssh infino-run-X.us-east1-b.pici-hammerlab`

Then update `env.hosts` setting in `fabfile.py`.

Run `fab --list` to make sure fabric is working.

# Configure tasks

Configure per-host tasks in `host_task_map.py` -- make sure to update `map_host_to_task` and `map_host_to_docker_name`. Which task is run depends entirely on the dictionary defined in that file, and the keys are host names. So I recommend putting details about the experiment into the host name. 

# Confirm connections: monitor diskspace

`fab -P diskspace`

# Install dependencies

`fab -P bootstrap_node`

(`-P` means [run in parallel](http://docs.fabfile.org/en/1.13/usage/parallel.html))

The sanity check can be hard to see in the logs because it's running in parallel across nodes. So just run it again and make sure each one has `0` outputted: `fab -- cat infino/out/test_remote_run.txt`

# Launch main computation

Verify data versions: `fab -- 'cd infino/deploy && ./verify_data.sh'` (should see all OKs)

Dry run: `fab run_infino`

Real run: `fab run_infino:0`

The output from that will just be a docker container ID.

Make sure docker containers are alive:
`fab -- docker ps`
 (anonymous task)

If something went wrong (maybe run `fab -- docker ps -a` to see exit code and more) and need to destroy the docker containers, run `fab clean_up_docker_containers:0` (again `:0` is to avoid dry run, so start without that parameter to understand what's going on), then retry.

Note: That clean-up will only work on already-dead containers. You can stop or kill containers with `fab docker_stop:0` or `fab docker_kill:0`, respectively.

View docker logs with:
`fab docker_logs`]
 (runs log commands customized with the name of the docker container running on each particular host)

Confirm that our logging to disk is working:
`fab -- 'tail -n 10 infino/out/*.consoleout.txt'`

Confirm that sampling logs are being written:
`fab -- 'ls -lh infino/out/*.samples.*.csv'`

# Monitor progress the same way

```
# make sure containers are alive
fab -- docker ps

# check sampling percentages
fab -- 'tail -n 10 infino/out/*.consoleout.txt'

# check out what stdout shows
fab -- 'tail -n 10 infino/out/*.stdout.*.txt'

# how many chains converged or otherwise completed?
fab -- 'grep "Execution completed" infino/out/*.stdout.*.txt | wc -l' | less

# what exit codes do we have?
fab -- 'grep "Execution completed" infino/out/*.stdout.*.txt' | less

# how did stansummary go?
fab -- 'grep "stansummary completed" infino/out/*.consoleout.txt' | less

# any errors?
fab -- 'grep -i "error" infino/out/*.consoleout.txt'

# check container exit codes
fab docker_container_exit_code | less
```

# Upload all results to gcloud storage

`fab upload_to_bucket:dryrun=0` (as usual, dry run it first)

Confirm this by using `gsutil` locally:

```
> gsutil ls -lh 'gs://infino-hammerlab/run_output/*/'

TODO
```

# Clean up

turn the instances down with:
```
gcloud compute instances stop infino-run-1 infino-run-2 infino-run-3;
```

TODO later clean up more fully with:

```
gcloud compute instances delete infino-run-1 infino-run-2 infino-run-3 --keep-disks all
# gcloud compute instances delete --delete-disks=all --quiet infino-run-1 infino-run-2 infino-run-3; # delete disks
```


# Download data

Go to the directory you want to download data to (MZ note: `/data/modelcache_new/experiments`), then run: `gsutil -m rsync -r gs://infino-hammerlab/run-output/ .`


# Diagnostic plots (TODO fix this up)

First, `mkdir -p plots && chmod -R 777 plots`

Ran a test with

```
docker run -it --rm \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--cohort_name bladder \
--cohort_name_cib newbladder \
--slug bladder_counts_raw \
--metric counts \
--processing raw \
--cutname cut1 \
--stansummary logs/bladder/stansummary.bladder.counts_raw.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_0.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_1.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_2.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_3.csv \
--noplots \
;"
```

Ran for real with:


```
docker run -it -d --name plot_bladder_counts_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--cohort_name bladder \
--cohort_name_cib newbladder \
--slug bladder_counts_raw \
--metric counts \
--processing raw \
--cutname cut1 \
--stansummary logs/bladder/stansummary.bladder.counts_raw.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_0.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_1.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_2.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_raw.txt_3.csv \
;"


docker run -it -d --name plot_bladder_counts_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--cohort_name bladder \
--cohort_name_cib newbladder \
--slug bladder_counts_corrected \
--metric counts \
--processing corrected \
--cutname cut1 \
--stansummary logs/bladder/stansummary.bladder.counts_corrected.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_corrected.txt_0.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_corrected.txt_1.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_corrected.txt_2.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_counts_corrected.txt_3.csv \
;"

docker run -it -d --name plot_bladder_tpm_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--cohort_name bladder \
--cohort_name_cib newbladder \
--slug bladder_tpm_corrected \
--metric tpm \
--processing corrected \
--cutname cut1 \
--stansummary logs/bladder/stansummary.bladder.tpm_corrected.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_corrected.txt_0.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_corrected.txt_1.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_corrected.txt_2.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_corrected.txt_3.csv \
;"

docker run -it -d --name plot_bladder_tpm_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--cohort_name bladder \
--cohort_name_cib newbladder \
--slug bladder_tpm_raw \
--metric tpm \
--processing raw \
--cutname cut1 \
--stansummary logs/bladder/stansummary.bladder.tpm_raw.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_raw.txt_0.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_raw.txt_1.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_raw.txt_2.csv \
--trace logs/bladder/sampling_log.bladder.cutbladder_tpm_raw.txt_3.csv \
;"


# TCGA

# have to use only the traces that are valid 

docker run -it -d --name plot_tcgakirc_cut1_counts_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut1_counts_raw/stansummary.tcgakirc_cut1_counts_raw.csv \
--slug tcgakirc_cut1_counts_raw \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric counts \
--processing raw \
--cutname cut1 \
--trace logs/tcgakirc_cut1_counts_raw/sampling_log.cohort.tcgakirc_cut1_counts_raw.txt_0.csv \
--trace logs/tcgakirc_cut1_counts_raw/sampling_log.cohort.tcgakirc_cut1_counts_raw.txt_2.csv \
--trace logs/tcgakirc_cut1_counts_raw/sampling_log.cohort.tcgakirc_cut1_counts_raw.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut1_tpm_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut1_tpm_raw/stansummary.tcgakirc_cut1_tpm_raw.csv \
--slug tcgakirc_cut1_tpm_raw \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric tpm \
--processing raw \
--cutname cut1 \
--trace logs/tcgakirc_cut1_tpm_raw/sampling_log.cohort.tcgakirc_cut1_tpm_raw.txt_0.csv \
--trace logs/tcgakirc_cut1_tpm_raw/sampling_log.cohort.tcgakirc_cut1_tpm_raw.txt_2.csv \
--trace logs/tcgakirc_cut1_tpm_raw/sampling_log.cohort.tcgakirc_cut1_tpm_raw.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut1_counts_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut1_counts_corrected/stansummary.tcgakirc_cut1_counts_corrected.csv \
--slug tcgakirc_cut1_counts_corrected \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric counts \
--processing corrected \
--cutname cut1 \
--trace logs/tcgakirc_cut1_counts_corrected/sampling_log.cohort.tcgakirc_cut1_counts_corrected.txt_1.csv \
--trace logs/tcgakirc_cut1_counts_corrected/sampling_log.cohort.tcgakirc_cut1_counts_corrected.txt_2.csv \
--trace logs/tcgakirc_cut1_counts_corrected/sampling_log.cohort.tcgakirc_cut1_counts_corrected.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut1_tpm_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut1_tpm_corrected/stansummary.tcgakirc_cut1_tpm_corrected.csv \
--slug tcgakirc_cut1_tpm_corrected \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric tpm \
--processing corrected \
--cutname cut1 \
--trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_0.csv \
--trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_1.csv \
--trace logs/tcgakirc_cut1_tpm_corrected/sampling_log.cohort.tcgakirc_cut1_tpm_corrected.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut2_counts_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut2_counts_raw/stansummary.tcgakirc_cut2_counts_raw.csv \
--slug tcgakirc_cut2_counts_raw \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric counts \
--processing raw \
--cutname cut2 \
--trace logs/tcgakirc_cut2_counts_raw/sampling_log.cohort.tcgakirc_cut2_counts_raw.txt_0.csv \
--trace logs/tcgakirc_cut2_counts_raw/sampling_log.cohort.tcgakirc_cut2_counts_raw.txt_1.csv \
--trace logs/tcgakirc_cut2_counts_raw/sampling_log.cohort.tcgakirc_cut2_counts_raw.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut2_tpm_raw \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut2_tpm_raw/stansummary.tcgakirc_cut2_tpm_raw.csv \
--slug tcgakirc_cut2_tpm_raw \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric tpm \
--processing raw \
--cutname cut2 \
--trace logs/tcgakirc_cut2_tpm_raw/sampling_log.cohort.tcgakirc_cut2_tpm_raw.txt_0.csv \
--trace logs/tcgakirc_cut2_tpm_raw/sampling_log.cohort.tcgakirc_cut2_tpm_raw.txt_1.csv \
--trace logs/tcgakirc_cut2_tpm_raw/sampling_log.cohort.tcgakirc_cut2_tpm_raw.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut2_counts_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut2_counts_corrected/stansummary.tcgakirc_cut2_counts_corrected.csv \
--slug tcgakirc_cut2_counts_corrected \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric counts \
--processing corrected \
--cutname cut2 \
--trace logs/tcgakirc_cut2_counts_corrected/sampling_log.cohort.tcgakirc_cut2_counts_corrected.txt_0.csv \
--trace logs/tcgakirc_cut2_counts_corrected/sampling_log.cohort.tcgakirc_cut2_counts_corrected.txt_2.csv \
--trace logs/tcgakirc_cut2_counts_corrected/sampling_log.cohort.tcgakirc_cut2_counts_corrected.txt_3.csv \
;"

docker run -it -d --name plot_tcgakirc_cut2_tpm_corrected \
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \
-v /data/modelcache_new/experiments:/home/jovyan/work/model-single-origin-samples/infino-rcc/logs:ro \
hammerlab/infino-docker:latest \
bash -c "cd work/model-single-origin-samples/infino-rcc/ && python analyze_cut.py \
--stansummary logs/tcgakirc_cut2_tpm_corrected/stansummary.tcgakirc_cut2_tpm_corrected.csv \
--slug tcgakirc_cut2_tpm_corrected \
--cohort_name tcgakirc \
--cohort_name_cib tcgakirc \
--metric tpm \
--processing corrected \
--cutname cut2 \
--trace logs/tcgakirc_cut2_tpm_corrected/sampling_log.cohort.tcgakirc_cut2_tpm_corrected.txt_0.csv \
--trace logs/tcgakirc_cut2_tpm_corrected/sampling_log.cohort.tcgakirc_cut2_tpm_corrected.txt_1.csv \
--trace logs/tcgakirc_cut2_tpm_corrected/sampling_log.cohort.tcgakirc_cut2_tpm_corrected.txt_2.csv \
;"

```

When they're done (check exit code with `docker ps --all | grep 'plot_'`), execute: `mv plots results && sudo chown -R maxim:maxim results`, then git commit.

