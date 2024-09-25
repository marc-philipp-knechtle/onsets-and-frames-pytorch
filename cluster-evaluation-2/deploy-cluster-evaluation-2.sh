kubectl delete job onsets-and-frames-evaluation-2

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-evaluation-2:0.1.0
docker build -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-evaluation-2/k8s-cluster-evaluation-2.yaml

watch kubectl get pods
