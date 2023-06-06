# Use the official Python image with conda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Update conda and install Python 3.9.16
#RUN conda update -n base conda && \
#    conda install python=3.9.16 -y

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Set the entrypoint command
CMD ["python", "app.py"]
