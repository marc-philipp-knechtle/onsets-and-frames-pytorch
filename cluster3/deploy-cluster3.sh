kubectl delete job onsets-and-frames-pytorch-training-cluster3

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-cluster3:0.0.1
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster3/k8s-cluster3.yaml

kubectl get pods --watch