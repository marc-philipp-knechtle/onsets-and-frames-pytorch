git push

kubectl delete job onsets-and-frames-transcription

ssh "vingilot" "cd /home/ext/ch1/studentMPK/cluster-transcription/onsets-and-frames-pytorch && git pull && exit"

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/onsets-and-frames-transcription:0.0.1
docker build -f cluster-training/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-transcription/k8s-onsets-and-frames-transcription.yaml

watch kubectl get pods
