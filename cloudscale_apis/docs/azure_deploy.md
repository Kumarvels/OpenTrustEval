# Azure Deployment Script Example

## Azure App Service (Container)
- Build and push Docker image:
```bash
az acr build --registry <acr-name> --image opentrusteval:latest .
```
- Deploy to App Service:
```bash
az webapp create --resource-group <rg> --plan <plan> --name <app-name> --deployment-container-image-name <acr-name>.azurecr.io/opentrusteval:latest
```

## Azure Functions
- Use [Azure Functions Python](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python) for HTTP triggers.

## Secrets Management
- Use Azure Key Vault for API keys, DB passwords, etc.
- Reference secrets in App Service/Functions environment variables.

See Azure docs for full setup and security best practices.
