# kubectl delete all --all -n extch1
kubectl delete job onsets-and-frames-pytorch-training


export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-pytorch:0.0.1
docker build -f cluster1/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster1/k8s-job.yaml

kubectl get pods --watch