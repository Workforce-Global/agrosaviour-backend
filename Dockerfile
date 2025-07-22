# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container to the root
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code to the working directory
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Set environment variable
ENV PORT=8080

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
