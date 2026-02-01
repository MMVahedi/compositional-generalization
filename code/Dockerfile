# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PyTorch (if you need GPU support, you may need to adjust this)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . /app/

# Expose the application port (if needed)
EXPOSE 5000

# Define the command to run the application
CMD ["python", "main.py"]
