```
# preemptible (deletes in 24hr) test
gcloud beta compute --project "pici-hammerlab" instances create "infino-1" \
--zone "us-east1-b" --machine-type "n1-standard-1" --subnet "default" --no-restart-on-failure --maintenance-policy "TERMINATE" --preemptible --service-account "infino@pici-hammerlab.iam.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/cloud-platform" --min-cpu-platform "Automatic" --image "ubuntu-1604-xenial-v20171026a" --image-project "ubuntu-os-cloud" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "infino-1"


# real
gcloud beta compute --project "pici-hammerlab" instances create "infino-1" \
--zone "us-east1-b" --machine-type "n1-highmem-4" --subnet "default" --no-restart-on-failure \
--maintenance-policy "TERMINATE" --service-account "infino@pici-hammerlab.iam.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/cloud-platform" --min-cpu-platform "Automatic" --image "ubuntu-1604-xenial-v20171026a" --image-project "ubuntu-os-cloud" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "infino-1"
```

made service account in console that has full access to all

```
gcloud compute config-ssh # this makes a new ssh key, follow the prompts
# need to redo that after launching new instance

ssh infino-1.us-east1-b.pici-hammerlab
gsutil mb gs://infino-hammerlab/
```


turn down an instance:
```
# gcloud compute instances stop example-instance-1 example-instance-2
gcloud compute instances stop infino-1
#gcloud compute instances delete infino-1 --keep-disks all
gcloud compute instances delete infino-1 


```