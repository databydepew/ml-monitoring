
eval $(minikube docker-env) && docker build -t loan-model-api:latest .

kubectl apply -f manifests/deployment.yaml
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts && helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring -f prometheus-values.yaml

kubectl apply -f service-monitor.yaml
kubectl port-forward -n monitoring svc/loan-model-api 8001:8001
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 & kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &