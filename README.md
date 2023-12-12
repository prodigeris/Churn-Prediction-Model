# Churn Prediction Service

This Flask server is part of a churn prediction application that uses machine learning to predict customer churn based on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. It employs pre-trained models (`dt_model.pkl` for Decision Tree and `logistic_model.pkl` for Logistic Regression) to perform the predictions.

## Installation

To get the server up and running, you'll need to set up the environment and install dependencies using Conda:

1. Ensure that [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is installed on your machine.

2. Navigate to the repository's root directory in your terminal.

3. Use the provided `environment.yml` file to create a new Conda environment:

   ```shell
   conda env create -f environment.yml
   ```

4. Activate the new environment with:

   ```shell
   conda activate ChurnPrediction
   ```

## Running the Server

After setting up the environment, start the Flask server by:

1. Running the server script in the activated Conda environment:

   ```shell
   python server.py
   ```

2. By default, the Flask server will run on `http://127.0.0.1:5000/`.

## Usage

The server's index page provides access to the OpenAPI schema with detailed information on the endpoints and how to interact with the API. To predict customer churn, you can send a POST request to the server's endpoint with the appropriate customer data in JSON format.

For a detailed guide on the data schema, example requests, and to understand how the models were trained, refer to the [Jupyter notebook](./notebook.ipynb) provided in this repository.