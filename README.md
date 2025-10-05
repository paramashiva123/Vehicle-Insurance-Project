# 🚗 Health Insurance Cross-Sell Prediction 🏥

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![Cloud](https://img.shields.io/badge/Cloud-AWS-orange.svg)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-lightgrey.svg)
![Database](https://img.shields.io/badge/Database-MongoDB-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

This repository contains a complete, end-to-end MLOps project designed to predict which existing **Health Insurance** customers will be interested in purchasing **Vehicle Insurance**. The project showcases a full machine learning lifecycle, from initial data ingestion and model development to fully automated CI/CD deployment on the cloud.

---

## 📖 About the Project & Dataset

### Business Context
The client for this project is an insurance company that has provided Health Insurance to its customers. They need a model to predict whether these existing policyholders will also be interested in Vehicle Insurance offered by the company.

Building such a model is extremely valuable because it allows the company to plan its communication strategy effectively. By reaching out specifically to customers who are likely to be interested, the company can optimize its business model and increase revenue.

### Dataset Information
To predict a customer's interest in Vehicle Insurance, the model uses a dataset containing various customer attributes:
* **Demographics**: Gender, Age, Region Code.
* **Vehicle Details**: Vehicle Age, whether the vehicle has existing Damage.
* **Policy Information**: The annual premium paid for Health Insurance, the sourcing channel, etc.

---

## ✨ Key Features

This project is built with a focus on automation, scalability, and best practices in MLOps.

* **Automated ML Pipeline**:
    * A modular training pipeline (`src/pipline/training_pipeline.py`) that orchestrates every step: Data Ingestion, Validation, Transformation, Model Training, Evaluation, and Pushing.
    * Each step is encapsulated in its own class (e.g., `DataIngestion`, `DataValidation`) for maintainability.

* **CI/CD Automation**:
    * A full CI/CD workflow is configured using **GitHub Actions** (`.github/workflows/aws.yaml`).
    * Every `git push` to the main branch triggers the workflow, which automatically builds a **Docker** image, pushes it to **AWS Elastic Container Registry (ECR)**, and deploys the new version to an **AWS EC2** instance.
    * Utilizes a **self-hosted runner** configured on the EC2 instance to connect GitHub with the deployment server.

* **Cloud Integration & Model Registry**:
    * Leverages **AWS S3** as a robust model registry to store the production-ready model artifact.
    * This setup allows for seamless versioning and retrieval of the best-performing model for inference and evaluation.

* **Interactive Web Application & API**:
    * A user-friendly web interface built with **FastAPI** and **Jinja2** templates allows for real-time predictions by submitting a form.
    * Provides two main endpoints in `app.py`:
        * `POST /`: Accepts customer data from the web form and returns a prediction on their interest in **Vehicle Insurance**.
        * `GET /train`: Triggers the entire machine learning training pipeline on demand.

* **Champion-Challenger Model Evaluation**:
    * The pipeline implements a "Champion-Challenger" strategy where the newly trained model (challenger) is rigorously compared against the current production model (champion) from S3.
    * The comparison is based on **F1-score**. The new model is promoted to production only if its score is higher, ensuring continuous performance improvement.

* **Advanced Data Preprocessing**:
    * The `DataTransformation` class handles a multi-step preprocessing pipeline.
    * **Handles Class Imbalance**: Uses the `SMOTEENN` technique to oversample the minority class and clean the majority class, creating a more balanced dataset for training.
    * **Feature Engineering**: Maps `Gender` to binary, creates dummy variables for categorical features (`Vehicle_Age`, `Vehicle_Damage`), and renames columns for consistency.
    * **Feature Scaling**: Applies `StandardScaler` to numerical columns and `MinMaxScaler` to select features using a `ColumnTransformer` for robust preprocessing.

* **Scalable Data Storage**:
    * Uses **MongoDB Atlas**, a cloud-native NoSQL database, as the primary data source.
    * The `data_ingestion` component fetches data directly from the MongoDB collection, making the pipeline scalable to large datasets.

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

1.  **Data Ingestion**: The `DataIngestion` class connects to MongoDB, exports the collection as a DataFrame, and splits it into training and testing CSV files.
2.  **Data Validation**: The `DataValidation` class reads the train/test CSVs and validates them against a predefined schema (`config/schema.yaml`), ensuring data integrity.
3.  **Data Transformation**: The `DataTransformation` class applies all preprocessing steps, including scaling, encoding, and handling class imbalance with `SMOTEENN`.
4.  **Model Training**: The `ModelTrainer` class trains a `RandomForestClassifier` on the transformed data and bundles the preprocessor and model into a single artifact.
5.  **Model Evaluation**: The `ModelEvaluation` class fetches the current production model from AWS S3 and compares its F1-score against the newly trained model on the test set.
6.  **Model Pusher**: If the new model is accepted, the `ModelPusher` class uploads the new model artifact to the AWS S3 bucket, making it the new production model.
7.  **CI/CD Deployment**: When changes are pushed to GitHub, the workflow builds the Docker image, pushes it to AWS ECR, and the self-hosted runner on EC2 deploys the new container.
8.  **Prediction Service**: The running Docker container serves the FastAPI application, allowing users to get real-time predictions on whether a customer is interested in Vehicle Insurance.

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
    You must set the following environment variables for the application to connect to external services.
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