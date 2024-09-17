kubectl delete job onsets-and-frames-pytorch-training-cluster4

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-cluster4:0.0.4
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster4/k8s-cluster4.yaml

kubectl get pods --watch
