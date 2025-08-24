Prerequisites
Python 3.8+

Git

Docker

GCP SDK (gcloud)

Kubernetes CLI (kubectl)

wrk load generator

(Recommended) Python virtual environment

Setup
Clone Repository

text
git clone https://github.com/anuzzzzz/21F2000400_IITMBS_MLOPS_OPPE2.git
cd 21F2000400_IITMBS_MLOPS_OPPE2
Install Requirements

text
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Train Model

text
python src/model/train.py --data data/data-3.csv
GCP and Kubernetes Deployment
Build Docker Image

text
docker build -t gcr.io/<your-gcp-project-id>/heart-disease-api:latest -f docker/Dockerfile .
docker push gcr.io/<your-gcp-project-id>/heart-disease-api:latest
Create GCS Bucket (for model/data artifacts)

text
gsutil mb gs://<your-mlops-bucket>
gsutil cp models/model.pkl gs://<your-mlops-bucket>/models/
Deploy to GKE (Google Kubernetes Engine)

text
# Create cluster if not done already
gcloud container clusters create ml-cluster --num-nodes=2 --zone=<your-zone>
gcloud container clusters get-credentials ml-cluster --zone=<your-zone>

# Deploy app and expose with LoadBalancer
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
Autoscaling

Horizontal Pod Autoscaler is set with maxReplicas: 3 (see k8s/deployment.yaml).

Objective Implementations
Explainability: Feature Importance
Used SHAP to identify key predictors for heart disease.

Top contributing factors (per SHAP analysis):

cp (chest pain type), thalach (max heart rate), oldpeak, exang, age

See /explainability/shap_explanation.png and /fairlearn_analysis/fairness_report.md

Fairness: Sensitive Attribute gender (Fairlearn)
Model fairness across gender evaluated using Fairlearn.

Metrics: disparate impact, demographic parity difference.

Results and figures under /fairlearn_analysis/.

Dockerized Model API on GCP, k8s Autoscaling (max pod-3)
API framework: [Flask/FastAPI] for model inference (src/api/app.py)

Docker: Containerizes full API stack.

K8s: Deployment script with autoscaler in /k8s/

GCP: Hosted on Google Kubernetes Engine, model weights pulled from GCS at pod startup.

Per-Sample Prediction, Logging, Observability
API endpoint supports single-record /predict.

All requests/responses logged (see /logs/prediction_logs/).

100 synthetic samples generated and used for demo, saved at /random_samples/synthetic_100.csv.

Performance Monitoring & Timeout (wrk)
wrk used for high-load request testing (see /wrk/).

Scripts and reports: average latency, timeout error rates, and throughput tested under concurrent load.

Results documented in /wrk/wrk_report.md.

Input Drift Detection
Input data distribution shift measured using statistical comparison (Kolmogorov-Smirnov test/JSD) between train and generated prediction samples.

Drift analysis, metrics, and visuals are in /drift/drift_report.md.

Data Poisoning Attack
Simulated by interchanging positive and negative labels in a subset of the data.

Model performance (e.g., accuracy, ROC-AUC) compared before and after.

Results available in /poisoning/poisoned_performance_comparison.md.

How to Use the Model API
Predict via API
text
curl -X POST "<your-service-endpoint>/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 55, "gender": "male", ... }'
Example requests and OpenAPI schema are available in /src/api/.

Observability & Monitoring
All predictions and errors are timestamp-logged.

Key metrics (traffic, error rate, latency) viewable via GCP console (Stackdriver).

Logs stored in /logs/ and also sent to GCP’s logging pipeline.

Performance & Input Drift Reports
See /wrk/, /drift/ subfolders for full markdown summaries, plots, and test methodology.

Data Artifacts
Training data: /data/data-3.csv

Synthetic eval data: /random_samples/synthetic_100.csv

Model artifact: /models/model.pkl (also uploaded to GCS)

Additional Notes
For full experiment reproducibility, reference major commit hashes for each milestone in your history.

Please see the inline comments and commit messages for integration insights.

Contact
For evaluation, queries, or repo access issues, contact

da5014_1@study.iitm.ac.in (course staff)

End of README
(Adapt, fill in your actual resource names, file paths, and any missing URLs as needed before final submission.)