# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for database connection
ENV DB_HOST_IP=172.17.0.2
ENV DB_PORT=3306
ENV DB_USER=root
ENV DB_PASSWORD=*****
ENV DB_NAME=Mesonet\ Data


# Run random_forest.py when the container launches
CMD ["python", "./random_forest.py"]

