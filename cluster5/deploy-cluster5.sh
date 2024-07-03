kubectl delete job onsets-and-frames-pytorch-training-cluster5

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-cluster5:0.0.1
docker build --no-cache -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster5/k8s-cluster5.yaml

kubectl get pods --watch
