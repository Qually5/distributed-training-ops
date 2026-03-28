#!/bin/bash

# distributed-training-ops/setup_cluster.sh
# This script automates the setup of a Kubernetes cluster for distributed ML training.

set -euo pipefail

KUBECONFIG_PATH="$HOME/.kube/config"
CLUSTER_NAME="ml-training-cluster"
REGION="us-central1"
NODE_TYPE="n1-standard-8"
NUM_NODES="3"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

install_gcloud_cli() {
  if ! command -v gcloud &> /dev/null
  then
    log "gcloud CLI not found. Installing..."
    sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install -y google-cloud-sdk
    log "gcloud CLI installed."
  else
    log "gcloud CLI already installed."
  fi
}

authenticate_gcloud() {
  log "Authenticating with Google Cloud..."
  # This would typically involve `gcloud auth login` and `gcloud config set project`
  # For automation, service account key is often used.
  # Assuming gcloud is already configured or service account is set up in the environment.
  gcloud auth list
  gcloud config set project $(gcloud config get-value project)
  log "Google Cloud authenticated."
}

create_kubernetes_cluster() {
  log "Checking for existing Kubernetes cluster: $CLUSTER_NAME in $REGION..."
  if gcloud container clusters list --filter="name=$CLUSTER_NAME" --zone="$REGION" --format="value(name)" | grep -q "$CLUSTER_NAME"; then
    log "Cluster $CLUSTER_NAME already exists. Skipping creation."
  else
    log "Creating Kubernetes cluster: $CLUSTER_NAME in $REGION with $NUM_NODES $NODE_TYPE nodes..."
    gcloud container clusters create $CLUSTER_NAME       --zone $REGION       --machine-type $NODE_TYPE       --num-nodes $NUM_NODES       --enable-stackdriver-kubernetes       --enable-ip-alias       --no-enable-basic-auth       --no-issue-client-certificate       --enable-autoupgrade       --enable-autorepair       --workload-pool=$(gcloud config get-value project).svc.id.goog       --project=$(gcloud config get-value project)
    log "Kubernetes cluster $CLUSTER_NAME created successfully."
  fi
  
  log "Configuring kubectl to connect to $CLUSTER_NAME..."
  gcloud container clusters get-credentials $CLUSTER_NAME --zone $REGION --project=$(gcloud config get-value project)
  log "kubectl configured."
}

install_nvidia_gpu_operator() {
  log "Installing NVIDIA GPU Operator (if GPUs are present and needed)..."
  # This is a placeholder. Actual GPU operator installation involves Helm charts and specific configurations.
  # kubectl create namespace gpu-operator
  # helm repo add nvidia https://nvidia.github.io/gpu-operator
  # helm install --wait nvidia/gpu-operator --generate-name -n gpu-operator
  log "NVIDIA GPU Operator installation simulated."
}

deploy_ml_training_job() {
  log "Deploying a sample distributed ML training job..."
  # This would involve applying Kubernetes YAML manifests for a distributed training job,
  # e.g., using Kubeflow Training Operator or raw Pods/Deployments.
  # Example: kubectl apply -f training-job.yaml
  
  cat <<EOF > training-job.yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: tfjob-simple-example
spec:
  tfReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.10.0-gpu
              command:
                - python
                - -c
                - |-
                  import tensorflow as tf
                  import os
                  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
                  strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver)
                  with strategy.scope():
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
                        tf.keras.layers.Dense(10, activation='softmax')
                    ])
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                  
                  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                  x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
                  y_train = y_train.astype('int32')
                  
                  # Limit dataset size for quick demo
                  x_train = x_train[:1000]
                  y_train = y_train[:1000]

                  model.fit(x_train, y_train, epochs=3)
                  print("Distributed training job completed successfully!")
EOF
  
  kubectl apply -f training-job.yaml
  log "Sample ML training job deployed. Monitor with: kubectl get tfjob tfjob-simple-example -w"
}

monitor_cluster() {
  log "Setting up basic cluster monitoring..."
  # This would involve deploying Prometheus/Grafana or configuring Stackdriver/Cloud Monitoring.
  # For now, just show basic kubectl commands.
  log "To check node status: kubectl get nodes"
  log "To check deployed pods: kubectl get pods -n default"
  log "To check services: kubectl get services -n default"
  log "Monitoring setup simulated."
}

main() {
  log "Starting distributed ML training cluster setup..."
  install_gcloud_cli
  authenticate_gcloud
  create_kubernetes_cluster
  install_nvidia_gpu_operator # Optional, if GPUs are needed
  deploy_ml_training_job
  monitor_cluster
  log "Distributed ML training cluster setup complete!"
}

main
