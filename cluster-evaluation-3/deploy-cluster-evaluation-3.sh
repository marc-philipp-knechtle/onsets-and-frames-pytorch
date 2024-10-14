kubectl delete job onsets-and-frames-evaluation-3

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-evaluation-3:0.0.4
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-evaluation-3/k8s-cluster-evaluation-3.yaml

watch kubectl get pods
