# Vehicle Insurance Prediction Data Pipeline

## 🚀 Project Overview

This project is an end-to-end machine learning pipeline for predicting vehicle insurance outcomes. It follows modern MLOps practices with a modular architecture, cloud integration, and containerization for seamless deployment.

## 📂 Project Structure

```
src/
├── __init__.py
├── components/               # Pipeline components
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
├── configuration/            # Cloud and DB configurations
│   ├── __init__.py
│   ├── mongo_db_connection.py
│   └── aws_connection.py
├── cloud_storage/            # Cloud storage operations
│   ├── __init__.py
│   └── aws_storage.py
├── data_access/              # Data access layer
│   ├── __init__.py
│   └── proj1_data.py
├── constants/                # Project constants
│   └── __init__.py
├── entity/                   # Configuration and artifact entities
│   ├── __init__.py
│   ├── config_entity.py
│   ├── artifact_entity.py
│   ├── estimator.py
│   └── s3_estimator.py
├── exception/                # Custom exceptions
│   └── __init__.py
├── logger/                   # Logging configuration
│   └── __init__.py
├── pipline/                  # Training and prediction pipelines
│   ├── __init__.py
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── main_utils.py
config/                       # Configuration files
│   ├── model.yaml
│   └── schema.yaml
app.py                        # FastAPI application
requirements.txt              # Python dependencies
Dockerfile                    # Container configuration
.dockerignore                 # Docker ignore rules
demo.py                       # Demonstration script
setup.py                      # Package configuration
pyproject.toml                # Build system configuration
```

## 🛠️ Technology Stack

### ☁️ Cloud Services
- **AWS S3**: For model and artifact storage
- **AWS EC2**: For deployment and hosting
- **MongoDB Atlas**: For data storage and retrieval

### 🐍 Python Ecosystem
- **FastAPI**: For building the web application
- **Pydantic**: For data validation
- **PyMongo**: For MongoDB interactions
- **Boto3**: For AWS services integration

### 🚢 Containerization & Deployment
- **Docker**: For containerization
- **Docker Compose**: For multi-container management

### 📊 Machine Learning
- **Scikit-learn**: For model training and evaluation
- **PyYAML**: For configuration management

## 🔍 Key Features

1. **Modular Pipeline Architecture**: Each ML step (ingestion, validation, transformation, etc.) is encapsulated in separate components
2. **Cloud-Native Design**: Built from the ground up for cloud deployment with AWS integration
3. **Configuration Management**: YAML-based configuration for models and data schemas
4. **Containerized Deployment**: Ready for Docker-based deployment
5. **Comprehensive Logging**: Built-in logging system for tracking pipeline execution
6. **REST API Interface**: FastAPI-powered endpoints for predictions

## 🏗️ Pipeline Workflow

1. **Data Ingestion**: Pull data from MongoDB Atlas
2. **Data Validation**: Ensure data quality and schema compliance
3. **Data Transformation**: Prepare data for modeling
4. **Model Training**: Train machine learning models
5. **Model Evaluation**: Assess model performance
6. **Model Deployment**: Push validated models to AWS S3
7. **API Serving**: Make predictions available via FastAPI endpoints

## 🚀 Getting Started

1. Set up your AWS credentials and MongoDB Atlas connection
2. Configure the YAML files in the `config/` directory
3. Build and run the Docker container
4. Access the FastAPI application for training and predictions

This project demonstrates a production-grade ML pipeline following best practices for maintainability, scalability, and reliability in cloud environments.
