# Use the official Python 3.12 image for amd64
FROM --platform=linux/amd64 python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install SUMO
RUN apt-get update && apt-get install -y \
    sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/* \
    && echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

# Set SUMO_HOME environment variable
ENV SUMO_HOME=/usr/share/sumo

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
