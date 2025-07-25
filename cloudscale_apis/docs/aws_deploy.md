# AWS Deployment Script Example

## Lambda + API Gateway
- Use [Mangum](https://github.com/jordaneremieff/mangum) to run FastAPI on AWS Lambda.
- Example deployment (using AWS CLI):

```bash
# Package app
zip -r app.zip .
# Deploy Lambda
aws lambda create-function --function-name OpenTrustEvalAPI \
  --zip-file fileb://app.zip --handler mangum_handler.handler --runtime python3.12 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_LAMBDA_ROLE
# Create API Gateway and connect to Lambda (see AWS docs)
```

## ECS/EKS/EC2
- Build Docker image and push to ECR:
```bash
docker build -t opentrusteval .
aws ecr create-repository --repository-name opentrusteval
# Tag and push
docker tag opentrusteval:latest <account-id>.dkr.ecr.<region>.amazonaws.com/opentrusteval:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/opentrusteval:latest
```
- Deploy to ECS/EKS using Fargate or Kubernetes manifests.

## Secrets Management
- Use AWS Secrets Manager or SSM Parameter Store for API keys, DB passwords, etc.
- Reference secrets in Lambda/ECS environment variables.

See AWS docs for full setup and security best practices.
