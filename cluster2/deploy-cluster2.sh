export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-onsets-and-frames-cluster2:0.0.1
docker build -f cluster2/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f k8s-cluster2.yaml

kubectl get pods --watch