# Chest X-rays Classification

Welcome to my Chest X-ray Classification project! In this deep learning endeavor, I aim to analyze chest X-ray images and employ convolutional neural networks (CNNs) for accurate classification. Specifically, I'll be using transfer learning with Xception as the base model. This approach allows me to capitalize on the pre-trained features of Xception, enhancing the model's ability to classify chest X-ray images effectively.

This project is part of the [#mlzoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/) course, and its primary goal is to assist healthcare professionals in efficiently diagnosing chest-related illnesses, with a particular focus on conditions affecting the respiratory system.

The main objective is to leverage deep learning techniques for the precise classification of chest X-ray images into different disease categories. The model will undergo training using a diverse dataset that encompasses a spectrum of common chest diseases.

## Table of Contents
- [Data](#data)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Containerization](#containerization)
- [Cloud deployment](#Cloud-Deployment)

## Data

### Dataset

The dataset follows a simple directory structure, branching into "test" and "train" sets. Each set is further categorized into three classes, representing specific chest diseases. This organization aids in the seamless retrieval of images for training and testing purposes.

Classes
The dataset comprises images associated with three distinct chest conditions:

Class 1: COVID
Class 2: Viral Pneumonia
Class 3: Normal

You can find the dataset [here](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

## Prerequisites

To run this project, you will need to have the following installed:

- Python 3.11
- Pipenv (for managing the virtual environment and dependencies)
- Docker

## Installation

To get started, follow these steps:

1. Clone the repository.
2. Open a terminal and navigate to the project directory.
3. Run the following command to install the dependencies using Pipenv:

```bash
pipenv install
```

## Usage

### Notebook

In the `notebook.ipynb` file you'll find:
  - Exploratory data analysis
  - Model Selection and tuning

### Train model

To train the convolutional neural networks model use the `train.py` file provided in this repository. Running `train.py` will generate the following file:

- `models/xception_v4L_ep:79_val:0.951.h5`

#### Running the Training Script

```bash
pipenv run python train.py
```
After running the script, you will find the xception_v4L_ep:79_val:0.951.h5 file in the directory models.

### Serving the Model

To serve the model, use the Flask library. The `predict.py` file creates a server on port 9696. When a POST request is sent to the route '/predict' with the input data in the request body, it returns the model's prediction.

#### Running the Server

```bash
pipenv run python predict.py
```
Once the server is up and running, you can send a POST request to http://localhost:9696/predict with the input data to get predictions from the model.

 #### Documentation

 You can see the API documentation on the endpoint [`/apidocs`](http://localhost:9696/apidocs)

#### Example Usage

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "url": "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/covid.jpg"
}' http://localhost:9696/predict
```
This cURL request sends a POST request to the server running at http://localhost:9696/predict with a JSON body containing x-ray image url. The server returns a response of:
```JSON
{
  "Covid": 4.374717712402344,
  "Normal": 2.042759656906128,
  "Viral Pneumonia": -0.023531116545200348
}
```

#### Test urls

```JSON
{
  "normal": "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/normal.jpeg",
  "covid": "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/covid.jpg",
  "pneumonia": "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/pneumonia.jpeg"
}
```

## Containerization

To containerize this project, follow these steps:

1. Ensure you have Docker installed on your system. If not, you can [download it here](https://www.docker.com/get-started).

2. Open a terminal and navigate to the project directory.

3. Run the following command to build a Docker image for the project:

```bash
docker build -t xrays .
```

This command will use the Dockerfile provided in the repository to build an image named xrays based on the python:3.11-slim image.

4. Once the image is built, you can run it in a Docker container with the following command:

```bash
docker run -it --rm -p 9696:9696 xrays:latest
```

This will start a container running the project on port 9696.

## Cloud Deployment

### Publish a docker image to Google Container Registry

**Pre-requisites**
1. Docker installed
2. GCloud SDK
2. User or service account with access required to push to GCR.

- Docker Registry Login with Google Cloud

```bash
gcloud auth configure-docker
```

- Tagging Docker Image

```bash
docker tag xrays gcr.io/<GCP_PROJECT_ID>/xrays:1.0
```

- Pushing Docker Image to gcr.io

```bash
docker push gcr.io/<GCP_PROJECT_ID>/xrays:1.0 
```

### Deploy container with Google Cloud Run

```bash
gcloud run deploy xrays-app --image gcr.io/<GCP_PROJECT_ID>/xrays:1.0 --memory=2G --port=9696 --region us-central1 --platform managed --allow-unauthenticated --quiet
```

### URL

Service URL: [https://xrays-app-srblipudfq-uc.a.run.app](https://xrays-app-srblipudfq-uc.a.run.app)

#### Example Usage

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "url": "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/pneumonia.jpeg"
}' https://xrays-app-srblipudfq-uc.a.run.app/predict
```

This cURL request sends a POST request to the server running at https://xrays-app-srblipudfq-uc.a.run.app/predict with a JSON body containing x-ray image url. The server returns a response of:
```JSON
{
  "Covid":-0.42221733927726746,
  "Normal":-0.47069013118743896,
  "Viral Pneumonia":0.4358460009098053
}
```

 [Test urls](#Test-urls)

 #### Documentation

 You can see the API documentation on the endpoint [`/apidocs`](https://xrays-app-srblipudfq-uc.a.run.app/apidocs)