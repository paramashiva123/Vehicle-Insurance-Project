# 🚗 Vehicle Insurance Cross-Sell Prediction: An End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![Cloud](https://img.shields.io/badge/Cloud-AWS-orange.svg)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-lightgrey.svg)
![Database](https://img.shields.io/badge/Database-MongoDB-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

This repository contains a complete, end-to-end MLOps project designed to predict customer interest in a cross-sell campaign. The business goal is to identify existing vehicle insurance customers who are likely to be interested in purchasing health insurance. The project showcases a full machine learning lifecycle, from initial data ingestion and model development to fully automated CI/CD deployment on the cloud.

The core of this project is an automated pipeline that trains a classification model, evaluates its performance against a production model, and deploys it as a live API endpoint for real-time predictions.



---

## ✨ Key Features

This project is built with a focus on automation, scalability, and best practices in MLOps.

* **Automated ML Pipeline**:
    * A modular training pipeline (`src/pipline/training_pipeline.py`) that orchestrates every step: Data Ingestion, Validation, Transformation, Model Training, Evaluation, and Pushing.
    * Each step is encapsulated in its own class (e.g., `DataIngestion`, `DataValidation`) for maintainability.

* **CI/CD Automation**:
    * [cite_start]A full CI/CD workflow is configured using **GitHub Actions** (`.github/workflows/aws.yaml`)[cite: 21].
    * [cite_start]Every `git push` to the main branch triggers the workflow, which automatically builds a **Docker** image, pushes it to **AWS Elastic Container Registry (ECR)**, and deploys the new version to an **AWS EC2** instance[cite: 21, 22, 23, 27].
    * [cite_start]Utilizes a **self-hosted runner** configured on the EC2 instance to connect GitHub with the deployment server[cite: 25].

* **Cloud Integration & Model Registry**:
    * [cite_start]Leverages **AWS S3** as a robust model registry to store the production-ready model artifact[cite: 16, 17].
    * This setup allows for seamless versioning and retrieval of the best-performing model for inference and evaluation.

* **Interactive Web Application & API**:
    * [cite_start]A user-friendly web interface built with **FastAPI** and **Jinja2** templates allows for real-time predictions by submitting a form[cite: 1].
    * Provides two main endpoints in `app.py`:
        * [cite_start]`POST /`: Accepts customer data from the web form and returns a prediction ("Response-Yes" or "Response-No")[cite: 1].
        * [cite_start]`GET /train`: Triggers the entire machine learning training pipeline on demand[cite: 1].

* **Champion-Challenger Model Evaluation**:
    * [cite_start]The pipeline implements a "Champion-Challenger" strategy where the newly trained model (challenger) is rigorously compared against the current production model (champion) from S3[cite: 7].
    * The comparison is based on **F1-score**. [cite_start]The new model is promoted to production only if its score is higher, ensuring continuous performance improvement[cite: 7].

* **Advanced Data Preprocessing**:
    * [cite_start]The `DataTransformation` class handles a multi-step preprocessing pipeline[cite: 2].
    * [cite_start]**Handles Class Imbalance**: Uses the `SMOTEENN` technique to oversample the minority class and clean the majority class, creating a more balanced dataset for training[cite: 2].
    * [cite_start]**Feature Engineering**: Maps `Gender` to binary, creates dummy variables for categorical features (`Vehicle_Age`, `Vehicle_Damage`), and renames columns for consistency[cite: 2].
    * [cite_start]**Feature Scaling**: Applies `StandardScaler` to numerical columns and `MinMaxScaler` to select features using a `ColumnTransformer` for robust preprocessing[cite: 2].

* **Scalable Data Storage**:
    * [cite_start]Uses **MongoDB Atlas**, a cloud-native NoSQL database, as the primary data source[cite: 5].
    * [cite_start]The `data_ingestion` component fetches data directly from the MongoDB collection, making the pipeline scalable to large datasets[cite: 3, 6].

---

## 🔧 Tech Stack & Tools

| Category                    | Technologies & Tools                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Language & Core Libraries** | Python, Pandas, NumPy                                                                                                            |
| **Machine Learning** | Scikit-learn (for `RandomForestClassifier`, pipelines, metrics), Imbalanced-learn (for `SMOTEENN`)                                 |
| **Web Framework & API** | FastAPI, Uvicorn (ASGI Server), Jinja2 (Templating)                                                                                |
| **Database** | MongoDB (with MongoDB Atlas for cloud hosting)                                                                                   |
| **Cloud Provider** | **Amazon Web Services (AWS)** |
| **AWS Services** | **S3** (Model Registry), **EC2** (Deployment Server), **ECR** (Docker Image Registry), **IAM** (Secure Access Management)           |
| **DevOps & CI/CD** | Docker (Containerization), GitHub Actions (CI/CD Automation)                                                                     |

---

## 🚀 Project Workflow Explained

The project is architected as a series of sequential, automated steps that form the complete MLOps lifecycle.

1.  [cite_start]**Data Ingestion**: The `DataIngestion` class connects to MongoDB using the `Proj1Data` utility, exports the collection as a DataFrame, and splits it into training and testing CSV files[cite: 3].
2.  **Data Validation**: The `DataValidation` class reads the train/test CSVs and validates them against a predefined schema (`config/schema.yaml`). [cite_start]It checks for the correct number of columns and the presence of all required features, ensuring data integrity before processing[cite: 4, 10].
3.  **Data Transformation**: The `DataTransformation` class applies all preprocessing steps. It uses a `ColumnTransformer` pipeline to apply different scaling methods to different columns and handles class imbalance with `SMOTEENN`. [cite_start]The final preprocessor object is saved for later use in prediction[cite: 2].
4.  **Model Training**: The `ModelTrainer` class trains a `RandomForestClassifier` on the transformed data. [cite_start]It saves the final model object, which is a custom class that bundles the preprocessing pipeline and the trained model together (`src/entity/estimator.py`)[cite: 5, 12].
5.  **Model Evaluation**: The `ModelEvaluation` class fetches the current production model from AWS S3. It then uses the test dataset to compare the F1-score of the newly trained model against the production model. [cite_start]The result determines if the new model is accepted[cite: 7].
6.  [cite_start]**Model Pusher**: If the new model is accepted, the `ModelPusher` class uploads the new model artifact to the designated AWS S3 bucket, replacing the old production model[cite: 6].
7.  **CI/CD Deployment**: When changes are pushed to GitHub, the workflow defined in `.github/workflows/aws.yaml` is triggered. [cite_start]It builds the Docker image, pushes it to AWS ECR, and the self-hosted runner on the EC2 instance pulls this new image and restarts the container, deploying the updated application without manual intervention[cite: 21, 25, 26, 27].
8.  **Prediction Service**: The running Docker container serves the `app.py` FastAPI application. Users can interact with the web UI to get predictions, or developers can integrate with the API endpoints.

---

## ⚙️ Getting Started

To set up and run this project on your local machine, follow these instructions.

### Prerequisites
* Git
* Conda or Python 3.10+
* An AWS Account with AWS CLI configured locally
* A MongoDB Atlas Account with a database and collection set up

### Local Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create --name vehicle python=3.10 -y
    conda activate vehicle
    ```

3.  **Install Dependencies:**
    This command will install the packages from `requirements.txt` and also install your `src` directory as a local package.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    [cite_start]You must set the following environment variables for the application to connect to external services[cite: 8, 9, 14, 15].
    ```bash
    export MONGODB_URL="<your_mongodb_atlas_connection_string>"
    export AWS_ACCESS_KEY_ID="<your_aws_access_key>"
    export AWS_SECRET_ACCESS_KEY="<your_aws_secret_key>"
    export AWS_DEFAULT_REGION="us-east-1"  # Or your preferred region
    ```

5.  **Run the FastAPI Application:**
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    The application will now be running and accessible at **`http://127.0.0.1:8000`**.

6.  **Trigger the Training Pipeline (Optional):**
    To run the entire model training pipeline manually, navigate to **`http://127.0.0.1:8000/train`** in your browser or send a GET request to that URL.

---

## 📂 Project Structure

The project is organized into a modular structure that separates concerns and improves code reusability.

```
.
├── .github/workflows/        # GitHub Actions CI/CD pipeline configuration
├── config/                   # Configuration files (schema.yaml, model.yaml)
├── notebooks/                # Jupyter notebooks for EDA and experimentation
├── static/                   # CSS files for the web interface
├── src/                      # Main source code for the project
│   ├── components/           # Classes for each ML pipeline stage (ingestion, validation, etc.)
│   ├── configuration/        # Database and cloud connection managers
│   ├── constants/            # Project-wide constants and variables
│   ├── data_access/          # Utility for accessing data from MongoDB
│   ├── entity/               # Data classes for configuration and artifacts
│   ├── exception/            # Custom exception handling module
│   ├── logger/               # Custom logging setup
│   ├── pipeline/             # Training and prediction pipeline orchestrators
│   └── utils/                # Helper functions
├── templates/                # Jinja2 HTML templates for the web app
├── app.py                    # FastAPI application entry point
├── Dockerfile                # Defines the container for the application
├── requirements.txt          # List of Python dependencies
└── setup.py                  # Makes the `src` directory installable as a package
```