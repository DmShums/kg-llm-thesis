# How to use this repository

In order, to run each module or experiment sepparately from main pipeline go to `runners` module and execute python notebooks.


# How to run logmap

if doesn't exist:
```bash
mkdir output
```

```bash
docker run --rm \
  -v /Users/shuma/Desktop/dyplom:/workspace \
  -w /workspace \
  amazoncorretto:8-alpine \
  java -jar logmap/logmap-matcher-4.0.jar \
    MATCHER \
    file:data/anatomy/human-mouse/human.owl \
    file:data/anatomy/human-mouse/mouse.owl \
    output/ \
    true
```


in case you run large ontologies make you allocate enough ram to colima or different provider and pass following flag
```
"java", "-Xmx10g", "-jar", "logmap/logmap-matcher-4.0.jar",
```

check config before it

1) Stop Colima
```
colima stop
```

2) If your system has 16GB RAM:
```
colima start --memory 12 --cpu 6
```

3) Verify
```
docker run --rm alpine free -h
```

You want something like:
```
Mem: 11G total
```