# GitHub Actions CI/CD Secrets Management

## How to Use Secrets in CI/CD

1. **Add Secrets to GitHub Repository**
   - Go to your repository on GitHub.
   - Click on `Settings` > `Secrets and variables` > `Actions`.
   - Click `New repository secret` and add secrets (e.g., `API_KEY`, `DB_PASSWORD`, `AWS_ACCESS_KEY_ID`, etc.).

2. **Reference Secrets in Workflow YAML**
   - Use the `secrets` context in your workflow files:
     ```yaml
     env:
       API_KEY: ${{ secrets.API_KEY }}
       DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
       AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
       AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
     ```

3. **Inject Secrets into Application**
   - Use a step to write secrets to a `.env` file or pass as environment variables:
     ```yaml
     - name: Create .env file
       run: |
         echo "API_KEY=${{ secrets.API_KEY }}" >> .env
         echo "DB_PASSWORD=${{ secrets.DB_PASSWORD }}" >> .env
         echo "AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}" >> .env
         echo "AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}" >> .env
     ```
   - Or, pass secrets directly to your application as environment variables.

4. **Best Practices**
   - Never commit real secrets to the repository.
   - Use the provided `secrets_template.env` as a reference only.
   - Rotate secrets regularly and restrict access.

See the `ci.yml` and `deploy.yml` workflows for usage examples.
