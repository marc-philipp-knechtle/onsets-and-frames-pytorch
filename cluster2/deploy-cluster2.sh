kubectl delete job onsets-and-frames-pytorch-training-cluster2

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-cluster2:0.0.4
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster2/k8s-cluster2.yaml

kubectl get pods --watch