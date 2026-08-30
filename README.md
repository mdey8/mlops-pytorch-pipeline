# 🚀 MLOps PyTorch Pipeline & Kubernetes Deployment

An end-to-end Machine Learning Operations (MLOps) pipeline built with PyTorch and deployed on Kubernetes. This repository covers automated training execution, dataset mounting, model serving rollout, Horizontal Pod Autoscaling (HPA), and endpoint validation within a dedicated Kubernetes cluster environment.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 MLOPS KUBERNETES PIPELINE                         |
+-----------------------------------------------------------------------------------+

   +------------------------+             +----------------------------------+
   |   PyTorch Model Code   |  -------->  |      Kubernetes Training Job     |
   | (Dataset / Preprocess) |             |       (training-job-qz9hj)       |
   +------------------------+             +----------------------------------+
                                                           |
                                                           v
                                          +----------------------------------+
                                          |     Trained Model Artifacts      |
                                          |   (Persistent Volume / Config)   |
                                          +----------------------------------+
                                                           |
                                                           v
   +------------------------+             +----------------------------------+
   | Cluster Autoscaler     |  -------->  |     Model Serving Deployment     |
   |         (HPA)          |             |       (serving-deployment)       |
   +------------------------+             +----------------------------------+
                                                           |
                                                           v
                                          +----------------------------------+
                                          |        Kubernetes Service        |
                                          |        (serving-service)         |
                                          +----------------------------------+
                                                           |
                                                           v
                                          +----------------------------------+
                                          |       Port Forwarding (8081)     |
                                          |     curl.exe http://localhost    |
                                          +----------------------------------+
Prerequisites & Setup:
Container Runtime: Docker Desktop (with Kubernetes engine enabled)

CLI Tools: kubectl (v1.24+), git, curl.exe

Shell Environment: Windows PowerShell / Bash

ML Frameworks: PyTorch, Torchvision, OpenCV, Scikit-Learn

🚀 Setup & Execution Walkthrough
1. Create Cluster Namespace
Isolate all pipeline resources inside a dedicated namespace:
  kubectl apply -f k8s/namespace.yaml
  kubectl get ns ml-training

2. Apply Configuration & Volume Maps
Load application configuration and dataset volume mounts:
  kubectl apply -f k8s/configmap.yaml -n ml-training

3. Trigger & Verify Training Job
Launch the PyTorch model training job and confirm completion (1/1 COMPLETIONS):
  kubectl apply -f k8s/training-job.yaml -n ml-training

# Check job completion status
  kubectl get jobs -n ml-training

4. Deploy Model Server & HPA
Deploy the serving layer alongside autoscaling rules:

  kubectl apply -f k8s/serving-deployment.yaml -n ml-training
kubectl apply -f k8s/serving-service.yaml -n ml-training
kubectl apply -f k8s/hpa.yaml -n ml-training

# Confirm all pods are running (1/1 READY)
kubectl get pods -n ml-training

🧪 Endpoint Verification & Testing
To test the deployment locally via port-forwarding:

1. Establish Port Forwarding:
  # Forward container port 80 to local port 8081
kubectl port-forward pod/<ACTIVE_SERVING_POD_NAME> 8081:80 -n ml-training

2. Execute Endpoint Test (in a second terminal):
  curl.exe http://localhost:8081/

