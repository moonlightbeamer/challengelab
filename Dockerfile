FROM python:3.9
WORKDIR /app
COPY . .

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Configure the Google Cloud SDK
RUN gcloud init --console-only

# Get the Project ID
RUN PROJECT_ID=$(gcloud config get-value project)

# Export the Project ID as an environment variable
ENV PROJECT_ID=$PROJECT_ID

# ENV PROJECT_ID=acn-amex-account-poc-sandbox
ENV LOCATION=us-central1
RUN pip install gunicorn
RUN pip install -r requirements.txt
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app