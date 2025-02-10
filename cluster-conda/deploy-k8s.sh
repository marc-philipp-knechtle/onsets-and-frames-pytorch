# kubectl delete all --all -n extch1
echo "THIS DOES NOT WORK! There are various errors while using the conda environment :("
exit 0
kubectl delete job onsets-and-frames-pytorch-training-cluster1


export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-pytorch:0.0.1
docker build -f cluster1/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster1/k8s-job.yaml

kubectl get pods --watch