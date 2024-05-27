# Use a Python 3.9 base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . .

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install dependencies
RUN pip install gunicorn
RUN pip install -r requirements.txt

# Set environment variables
ENV PORT=8080

# Authenticate with Artifact Registry
# ARG GOOGLE_APPLICATION_CREDENTIALS
# RUN echo "$GOOGLE_APPLICATION_CREDENTIALS" > /.config/gcloud/application_default_credentials.json
RUN echo {gcloud config get project} > PROJECT_ID_ARRAY
ARG PROJECT_ID = PROJECT_ID_ARRAY[0]
RUN gcloud auth configure-docker us-central1-docker.pkg.dev
RUN gcloud services enable compute.googleapis.com aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com --project {PROJECT_ID}
# Build the image
# Change "us-central1" to your desired region
# Change "qwiklabs-gcp-02-bd257a57746c" to your project ID
# Change "cymbal-web-ui-app-repo" to your actual repository name

RUN BUILD_IMAGE="us-central1-docker.pkg.dev/qwiklabs-gcp-02-bd257a57746c/cymbal-web-ui-app-repo/cymbal-web-app1"

# Run gunicorn and expose the port
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "main:app"]

# Push the image to Artifact Registry
# RUN docker build . -t $BUILD_IMAGE
# RUN docker push $BUILD_IMAGE