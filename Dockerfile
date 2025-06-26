# Use the base image
FROM ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

# Set the working directory
WORKDIR /app

# Copy the Python files into the container
COPY main.py /app/

# Install dependencies from a local requirements.txt
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Run the Python script
CMD ["python", "-u", "main.py"]
