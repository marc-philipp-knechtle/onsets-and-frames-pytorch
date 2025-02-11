git push

kubectl delete job onsets-and-frames-training

ssh "vingilot" "cd /home/ext/ch1/studentMPK/onsets-and-frames-training/onsets-and-frames-pytorch && git pull && exit"

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/onsets-and-frames-training:0.0.2
docker build -f cluster-training/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-training/onsets-and-frames-training.yaml

watch kubectl get pods
