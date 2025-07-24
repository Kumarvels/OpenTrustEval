# Cloud Provider Integration Examples

## AWS Lambda + API Gateway
- Deploy FastAPI endpoints as Lambda functions using AWS Lambda Powertools or Mangum.
- Use API Gateway for HTTPS routing and authentication.
- Store logs/results in S3 or DynamoDB.

## Google Cloud Run / Functions
- Deploy FastAPI app as a container to Cloud Run for autoscaling.
- Use Google Cloud Functions for lightweight webhooks.
- Store logs/results in Google Cloud Storage or Firestore.

## Azure Functions / App Service
- Deploy endpoints as Azure Functions (Python HTTP trigger).
- Use Azure API Management for routing and security.
- Store logs/results in Azure Blob Storage or Cosmos DB.

## CI/CD Example (GitHub Actions)
- `.github/workflows/ci.yml` for lint, test, and build steps.
- `.github/workflows/deploy.yml` for cloud deployment (see provider docs for secrets setup).

See provider docs and the `tests/` folder for more details.
