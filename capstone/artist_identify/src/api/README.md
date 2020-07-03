
1. Deploy a model

In a new terminal

```
docker run -p 8501:8501 --mount type=bind,source="/Users/abmodi/2020/dleng/capstone/artist_identify/models/baseline/2020-03-10_21:28:35",target="/models/my_baseline" -e MODEL_NAME=my_baseline -t tensorflow/serving
```

Model deployed at 'http://localhost:8501/v1/models/my_baseline:predict'

2. Create a process to track Docker CPU usage stats

```
`while true; do docker stats --no-stream >> stats.txt; done` & echo $!
```

Note the output process id PID.

4. Run load test

```
make perf
```

Once it completes note the response time percentiles.

5. Kill dokcer stat tracker using the PID

```
kill -9 <PID>
```

6. Stop docker container

Get the container id using ps and then stop.

```
docker ps
docker stop <CONTAINER_ID>
```



