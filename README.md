# Project Name

This project provides a step-by-step guide to setting up, training, and running the application using Python, Docker, and Docker Compose.

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.10
- Docker
- Docker Compose

## Setup Instructions

### 1. Create a Virtual Environment

Run the following command to create a virtual environment:

`virtualenv venv -p python3.10`

## Activate the virtual environment:

`source path_to_venv/bin/activate`

### 2. Install Dependencies

Install the required dependencies using requirements.txt:

`pip3 install -r requirements.txt`

### 3. Train the Model

extract files inside data folder

To train the model, run the following command:

`python3 main.py --train`

### 4. Build and Run Docker Containers

Bring down any running containers:

`docker-compose down`

Build and run the containers:

`docker-compose up --build`

### 5. Access the Application

Once the Docker containers are running, you can access the application at:

`http://localhost:5588/`
