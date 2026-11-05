# Credit Risk Assessment using SHAP

A FastAPI application and HTML interface for estimating a loan applicant's probability of default with the supplied, calibrated scikit-learn model. The notebook contains the model-training workflow and SHAP examples.

> **Important:** This is an educational estimate, not a lending decision or financial advice. Do not submit real applicant data unless you have an appropriate privacy and compliance review. Model outputs may be inaccurate or biased.

## Features

- FastAPI prediction API with input validation and a health check.
- Existing calibrated model and saved classification threshold loaded from files beside the application.
- Responsive, same-origin HTML/CSS/JavaScript frontend.
- Notebook with training, evaluation, and SHAP visualizations.
- Render Blueprint configuration for deployment.

## Technology

Python 3.12, FastAPI, Pydantic, Uvicorn, pandas, NumPy, scikit-learn, XGBoost, joblib, SHAP (notebook only), HTML, CSS, and JavaScript.

## Project structure

```text
.
├── main.py
├── credit_risk_model.pkl
├── best_threshold.pkl
├── credit_risk_dataset.csv       # Used by the training notebook; not read by the API
├── Credit_Risk.ipynb
├── requirements.txt              # Production dependencies
├── requirements-notebook.txt     # Optional notebook/SHAP dependencies
├── runtime.txt
├── render.yaml
└── static/
    ├── index.html
    ├── script.js
    └── style.css
```

Both `.pkl` files are required at runtime and must be committed with the application. The dataset is only used by the notebook and is not required by the deployed API. Before publishing the CSV to GitHub, verify its redistribution license and confirm it contains no identifying or confidential applicant data; if it is not approved for publication, keep an approved local copy for notebook use and exclude the CSV from the upload.

## Install and run locally

Use Python 3.12.10 (the version declared in `runtime.txt`). From the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> for the interface, <http://127.0.0.1:8000/docs> for interactive API documentation, or <http://127.0.0.1:8000/health> for the health check. `python main.py` also starts Uvicorn and uses `PORT` if it is set (otherwise port 8000).

### Run in VS Code

1. Open this project folder in VS Code.
2. Select the Python 3.12 virtual environment (`.venv`) with **Python: Select Interpreter**.
3. In the integrated PowerShell terminal, activate the environment and run the install commands above.
4. Start the server with `uvicorn main:app --reload --host 127.0.0.1 --port 8000`.
5. Open the local URL in a browser. Use **Ctrl+C** in the terminal to stop the server.

To run the training notebook in VS Code, install its optional dependencies with `pip install -r requirements-notebook.txt`, select the same environment as the notebook kernel, and keep `credit_risk_dataset.csv` beside the notebook.

## API

### `GET /health`

Returns `{"status":"ok"}` when the application has loaded its model. Returns HTTP 503 if no model is ready.

### `POST /predict`

Accepts JSON with all 11 model input fields. Unknown fields, unsupported category values, non-finite numbers, and out-of-range values are rejected with HTTP 422.

Example request:

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

Response shape (probability and risk result vary by input):

```json
{
  "default_probability": 0.0,
  "default_prediction": 0,
  "threshold": 0.6951976431239382,
  "Result": "Low Risk"
}
```

The numeric response values above illustrate the schema only; they are not reported model metrics or a prediction for the example request. `default_prediction` is 1 when the returned probability meets or exceeds the saved threshold.

## Model, preprocessing, and SHAP

The API passes the 11 raw fields to the saved scikit-learn estimator in the order used by the training data. The serialized estimator includes its preprocessing pipeline; the API does not separately encode or scale inputs. `best_threshold.pkl` supplies the probability cutoff. The dataset contains missing employment-length and interest-rate values, so the bundled training pipeline must remain responsible for its learned preprocessing.

The notebook's threshold workflow was corrected to use the calibrated classifier's probabilities and the standard F1 formula. The checked-in `.pkl` files were deliberately left unchanged. The saved threshold loads as `0.6951976431239382`. The current model and threshold produced predictions on the notebook's reproducible 6,305-row test split, and that threshold exactly matches the F1-maximizing threshold on that split. This is evidence that the threshold was selected on the same test data, so metrics on that split are **not an unbiased final evaluation**. Re-run the corrected workflow using a separate validation split for threshold selection and reserve an untouched test set before reporting performance or replacing the artifacts together.

The notebook uses `shap.TreeExplainer` on the classifier from `xg_model` and its transformed test data to produce feature attributions and a waterfall plot for an individual test row. The saved API model is a calibrated classifier, so the notebook SHAP plots do not directly explain the deployed calibrated probability. They are not an explanation endpoint for live API requests; the production API currently returns a probability and threshold-based classification only. No model was retrained or replaced for deployment.

Scikit-learn must stay at the version used to serialize the model (`1.6.1`); loading a pickle with another version can fail or change behavior. XGBoost `3.2.0` is pinned because it successfully loaded and predicted with the supplied model in the verification environment; XGBoost `3.4.1` failed to deserialize its embedded booster. Loading with 3.2.0 emits XGBoost's warning about older-version serialized models, so the artifact should eventually be migrated by loading it in a verified compatible environment and exporting the booster in XGBoost's stable model format. Only load model files from a trusted source: Python pickle/joblib files can execute code when loaded.

No accuracy, precision, recall, or other performance metric is claimed here. Refer to the notebook's evaluation outputs and validate the model independently before relying on its predictions.

## Environment variables

| Variable | Required | Description |
|---|---:|---|
| `PORT` | No | Render supplies this port. Locally, the app defaults to `8000`. |
| `CORS_ORIGINS` | No | Comma-separated, explicit trusted web origins (for example `https://example.com`). Leave unset for the bundled same-origin frontend. Wildcard origins are rejected. |

## Upload to GitHub

Create an empty GitHub repository, then from the project directory:

```powershell
git init
git add .
git status
git commit -m "Prepare credit risk app for deployment"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
git push -u origin main
```

Check `git status` before committing. Do not add `.env` files, credentials, or applicant data. Keep `credit_risk_model.pkl` and `best_threshold.pkl` in the repository because the API requires them.

## Deploy to Render

1. Push the project to GitHub using the instructions above.
2. In Render, choose **New +** > **Blueprint** and connect the GitHub repository, or create a **Web Service** from that repository.
3. Use the repository root as the service root directory. `render.yaml` configures the service name, Python runtime, health check, and commands.
4. For a manually created Web Service, select **Python** and set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Leave `CORS_ORIGINS` unset for the included frontend, which is served by this same application. If a separate frontend is used, set it to that site's exact origin(s), comma-separated.
6. Deploy. Wait for the build and service health check to succeed.
7. Open `https://<your-service-name>.onrender.com/`, then verify `https://<your-service-name>.onrender.com/health` returns `{"status":"ok"}`.

Render's free web services may sleep when idle; the first request after inactivity can take longer. The Render build installs only production dependencies; SHAP and notebook tools are not needed to serve predictions.

## Troubleshooting

- **Model artifact not found:** Confirm both `.pkl` files are at the repository root and were not omitted from the GitHub push.
- **Unpickling or scikit-learn/XGBoost version error:** Use the pinned `requirements.txt` and Python version in `runtime.txt`; do not change scikit-learn or XGBoost without deliberately migrating and validating the saved model.
- **Build fails while resolving packages:** Check Render's build log and ensure the service uses Python 3.12.10 and the repository root. Retry after correcting the reported package or network failure.
- **Health check is unhealthy:** Review the service logs for the startup model-load error. The API intentionally fails startup rather than serving with missing or invalid model artifacts.
- **Prediction returns HTTP 422:** Send every required field with the documented numeric types and one of the listed category values. Verify that ratios are finite and between 0 and 1.
- **Browser shows service unreachable:** Check the Render service is live and `/health` responds. The included page uses relative same-origin API paths; if separately hosting a frontend, configure its API URL and set `CORS_ORIGINS` to that exact origin.
- **Notebook kernel cannot import packages:** Install `requirements-notebook.txt` into the selected kernel environment and restart the kernel.

Verification was performed on Windows with Python 3.12.10, scikit-learn 1.6.1, and XGBoost 3.2.0. The deployment runtime is Python 3.12.10. Although model loading and inference succeeded with those pins, XGBoost emitted a compatibility warning; Linux/Render startup and prediction still need to be verified after deployment. The threshold was selected on the notebook test split, so use an untouched evaluation set before relying on reported performance or classifications.
