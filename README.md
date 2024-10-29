# Health Detective
![Health Detective Logo](URL_of_your_image)

## Overview
Health Detective is an innovative medical prediction application that leverages advanced machine learning techniques. Utilizing a Retrieval-Augmented Generation (RAG) model, it combines document retrieval with a large language model (LLM) to provide accurate medical predictions based on user queries. The application aims to assist users in understanding potential medical outcomes based on symptoms and patient profiles.

## Features
- **Intelligent Medical Predictions**: Users can ask questions about medical conditions, and the application generates responses based on relevant patient data.
- **Customizable Interface**: Built with Streamlit, the application offers a user-friendly interface for seamless interaction.
- **Efficient Data Handling**: The application utilizes FAISS for efficient vector search and retrieval from a large dataset of medical information.

## Project Structure
The project is organized into several key components:
- **Ingestion**: Manages the loading of embeddings and the vector store, and provides the LLM-based answer generation functionality.
- **App**: The main application that interfaces with users and processes their queries.
- **Requirements**: Lists all necessary libraries for running the application.

### Getting Started

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- `pip` for managing packages

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Health_Detective
2. **Install Required Libraries**:
   Create a virtual environment (optional but recommended) and install the required packages:
   ```bash
   pip install -r req.txt
   
## Data Preparation
  Place the dataset file Disease_symptom_and_patient_profile_dataset.xlsx in the project root directory. This dataset is crucial for the 
  vector database and LLM predictions.
  
## Running the Application
1. **Ingestion**:
   Run the ingestion script to prepare the vector database:
    ```bash
   python ingestion.py
  This will load the embeddings and create a persistent vector store.

3. **Launch the Application**:
    Start the Streamlit app:
    ```bash
    streamlit run app.py
  The application should open in your web browser.
  
## Screenshots

## Usage
Enter your medical question in the provided input box and click "Search."
The application will process your request and display the generated response based on the retrieved medical information.
Each query generates a unique conversation ID for tracking and context maintenance.
Performance Metrics
Hit Rate Evaluation Score: 90.52
Mean Reciprocal Rank (MRR): 81.33
These metrics reflect the model's high accuracy and effectiveness in generating relevant responses based on user queries.

## Performance Metrics

- **Hit Rate Evaluation Score**: 90.52
- **Mean Reciprocal Rank (MRR)**: 81.33

These metrics reflect the model's high accuracy and effectiveness in generating relevant responses based on user queries.

## Acknowledgments
Thank you to the contributors of the libraries used in this project, including Langchain, Hugging Face Transformers, and Streamlit.
