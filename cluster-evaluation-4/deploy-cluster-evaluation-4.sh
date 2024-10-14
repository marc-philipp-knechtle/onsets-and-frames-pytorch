kubectl delete job onsets-and-frames-evaluation-4

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-evaluation-4:0.0.2
docker build -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-evaluation-4/k8s-cluster-evaluation-4.yaml

watch kubectl get pods