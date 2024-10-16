# Utility Tool for Business Problems (UTBP)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7.1-brightgreen)](https://www.python.org/downloads/)

Welcome to the **Utility Tool for Business Problems (UTBP)**! Created by **Ahmet Selçuk Arslan**, UTBP is a comprehensive web application designed to help businesses analyze their data, run scoring analyses, and derive actionable insights. This application leverages machine learning models to predict customer behaviors and optimize business strategies. With a single function, you can generate hundreds of machine learning models, and the best model is automatically saved in your working directory for further analysis (see CODE.ipynb).

Go check CODE.ipynb for a demo workflow. There are several custom functions written by me to help you make visualized cost calculations, and train hundreds of machine learning models.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Demo](#demo)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)

## Features

- **Data Upload**: Upload your dataset in CSV format for analysis.
- **Scoring Analysis**: Run predictive models on your data to score leads and customers.
- **Strategy Optimization**: Calculate and visualize optimal business strategies based on predictive scores.
- **Interactive Dashboard**: Use a Streamlit-based frontend to interact with the data and results.
- **API Access**: A FastAPI backend providing RESTful endpoints for data processing and predictions.

## Architecture

The UTBP application consists of two main components:

1. **Backend**: A FastAPI application that processes data, runs machine learning models (using H2O), and provides endpoints for prediction and strategy calculation.
2. **Frontend**: A Streamlit application that allows users to interact with the backend, upload datasets, configure parameters, and visualize results.

## Demo

### Initializing FastAPI:

https://github.com/user-attachments/assets/24a02208-7bd9-444e-8f54-613c1930fd34

### Using the app via Streamlit:

https://github.com/user-attachments/assets/007ba6f7-4c60-4540-a3e5-7baf77b2bbf5


## Installation

### Prerequisites

- **Python 3.7.1**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Virtual Environment**
- **Java 8 or higher**

### Clone the Repository

```bash
git clone https://github.com/FrenziedFlame123/UTBP.git
cd UTBP
```

### Backend Setup
**Create a Virtual Environment**

```bash
python -m venv venv
```

**Activate the Virtual Environment**

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

**Install Dependencies**

```bash
pip install -r requirements.txt
```
Ensure that requirements.txt includes all necessary packages such as FastAPI, Uvicorn, H2O, pandas, numpy, etc.

**Prepare the Machine Learning Model**

Check CODE.ipynb for the full workflow that can help you to use the custom functions and create machine learning models.

**Run the Backend Server**

```bash
uvicorn backend_fastapi:app --reload --port 8000
```
The backend API should now be running at http://localhost:8000.

### Frontend Setup

Ensure you're still in the virtual environment. Then run the Frontend Application

```bash
streamlit run frontend_streamlit.py
```
The Streamlit app should open in your default web browser or be accessible at http://localhost:8501.

## Usage

### Access the Frontend

Open the Streamlit app in your browser (usually at http://localhost:8501).

### Upload Your Dataset

Click on the "Browse files" button to upload your CSV dataset.
The dataset should contain the necessary columns expected by the application.

Optionally, check the "Show raw data" box to preview the first 10 rows of your dataset.

Enter the average sales per period, periodic sales safeguard percentage, and other parameters as prompted.

Click on the "Run Analysis" button.
The application will communicate with the backend to process the data, run the model, and calculate the strategy.

The strategy summary and expected value plot will be displayed.
You can download the scoring strategy as a CSV file.
