kubectl delete job onsets-and-frames-evaluation

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-evaluation:0.2.4
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-evaluation/k8s-cluster-evaluation.yaml

watch kubectl get pods
