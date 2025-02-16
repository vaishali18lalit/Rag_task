# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required ports
EXPOSE 8000 8501

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor

# Copy the Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run Supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
