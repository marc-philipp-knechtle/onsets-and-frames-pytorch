kubectl delete all --all -n extch1

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-pytorch:0.0.1
docker build . -t $NAME
docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f k8s-job.yaml

kubectl get pods --watch