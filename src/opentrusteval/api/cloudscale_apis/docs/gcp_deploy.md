# GCP Deployment Script Example

## Cloud Run (Recommended)
- Build and push Docker image:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/opentrusteval
```
- Deploy to Cloud Run:
```bash
gcloud run deploy opentrusteval --image gcr.io/$PROJECT_ID/opentrusteval --platform managed --region us-central1 --allow-unauthenticated
```

## Cloud Functions
- Use [functions-framework](https://github.com/GoogleCloudPlatform/functions-framework-python) for HTTP triggers.

## Secrets Management
- Use Secret Manager for API keys, DB passwords, etc. Example:
```bash
gcloud secrets create opentrusteval-api-key --data-file=api_key.txt
gcloud secrets versions access latest --secret=opentrusteval-api-key
```
- Reference secrets in Cloud Run/Functions environment variables.

See GCP docs for full setup and security best practices.
