# BD24_Project_A4_A - Federated Sentiment Analysis Quickstart Local

This file contains a tutorial for the local quickstart of the Federated Sentiment Analysis System.

## Installations

Docker needs to be installed for this to work. Anything else will be installed in-container.

## Commands for Quickstart
Open the project in a terminal. If you are using a linux shell, enter
```cmd
cd flink && sudo docker build . --tag ml_client_taskmanager:latest && cd ../dataprovider && sudo docker build . --tag data-provider:latest && cd .. && sudo docker compose up
```
. You will have to supply the sudo password. This will start the network and have the containers connect. The build-process might take up to 10 minutes.

If you are on Windows, do
```cmd
cd flink && docker build . --tag ml_client_taskmanager:latest && cd ../dataprovider && docker build . --tag data-provider:latest && cd .. && docker compose up
```
.

Next, you will have to post the jobs to the Jobmanager. For this, open a new terminal per job and enter these commands on Linux:

```cmd
sudo docker exec jobmanager /opt/flink/bin/flink run \
        --python /tmp/src/flink_split_data_job.py
```

and these on Windows:
```cmd
docker exec jobmanager /opt/flink/bin/flink run \
        --python /tmp/src/flink_split_data_job.py
```


If everything worked, you will see messages being displayed in the terminal you called docker compose in.