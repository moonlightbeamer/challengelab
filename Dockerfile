FROM python:3.9
WORKDIR /app
COPY . .

# Download and install gcloud package
RUN curl -sSL https://sdk.cloud.google.com | bash

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN gcloud init

# get the project id
RUN export PROJECT_ID=$(gcloud config get-value project)
ENV LOCATION="us-central1"
RUN pip install gunicorn
RUN pip install -r requirements.txt
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app