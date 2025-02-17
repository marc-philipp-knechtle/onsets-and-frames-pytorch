git push

kubectl delete job onsets-and-frames-transcription-2

ssh "vingilot" "cd /home/ext/ch1/studentMPK/cluster-transcription-2/onsets-and-frames-pytorch && git pull && exit"

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/onsets-and-frames-transcription-2:0.0.1
docker build -f cluster-training/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster-transcription-2/k8s-onsets-and-frames-transcription-2.yaml

watch kubectl get pods
