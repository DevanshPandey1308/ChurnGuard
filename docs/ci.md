# ChurnGuard continuous integration

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes to
the repository's current default branch (`master`) and on pull requests. It
uses Ubuntu with Python 3.11, matching the inference Docker image.

CI installs the package's `analysis` and `test` extras, then runs pytest,
`compileall`, patch whitespace checks, `docker compose config`, and a build of
the existing Docker image. The `test` extra declares pytest; analysis packages
remain separate from the serving dependencies used by the image.

CI does not invoke the customer-data experiment pipeline or train/evaluate the
production churn and value models. One existing SHAP unit test fits a tiny
synthetic LightGBM fixture. CI does not download the UCI dataset or perform
inference against real model files, and it does not need notebooks, Power BI
files, MLflow services, external databases, or committed `models/` artifacts.
The model directory is excluded from the Docker build context and is only a
read-only runtime mount in local Compose. Compose configuration validation and
image building do not start the service or resolve/load the runtime model mount.

The workflow validates code and packages the inference image; it does not
publish or deploy the image and does not provide CI/CD release automation.
