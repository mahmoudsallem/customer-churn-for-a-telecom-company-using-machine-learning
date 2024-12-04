# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]