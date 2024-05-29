# PYTHON SCRIPT TO BUILD THE VECTOR SEARCH DATABASE

print("Python Script Started")

import IPython
from IPython.display import Markdown, display
import time
import os

PROJECT_ID = os.getenv("project_id")
LOCATION = os.getenv("lab_region")

from google.cloud import storage	

# Create a client	
client = storage.Client()	

# List the buckets	
buckets = list(client.list_buckets())	

# Check if there are any buckets	
if buckets:	
    # Store the name of the first bucket into a variable	
    BUCKET_NAME = buckets[0].name	
    print(BUCKET_NAME)	
else:	
    print("No buckets found.")

# generate an unique id for this session
from datetime import datetime

UID = datetime.now().strftime("%m%d%H%M")

import requests

URL = 'https://www.nyc.gov/assets/doh/downloads/pdf/rii/fpc-manual.pdf'

response = requests.get(URL, stream=True)

if response.status_code == 200:
  # Download the PDF content in chunks
  pdf_content = b''
  for chunk in response.iter_content(1024):
    pdf_content += chunk
else:
  print(f"Error: Could not download PDF from URL: {response.status_code}")

# Use the downloaded content with PyPDFLoader
with open('fpc-manual.pdf', 'wb') as f:
  f.write(pdf_content)

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("fpc-manual.pdf")  # Name the blob in the bucket
blob.upload_from_string(pdf_content)

print(f"PDF uploaded to Cloud Storage: gs://{BUCKET_NAME}/fpc-manual.pdf")

# convert pages array into an array of page_content
from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader('fpc-manual.pdf')
pages = pdf.load_and_split()

pages = [page.page_content for page in pages]

from typing import Generator, List, Optional, Tuple
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import math

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Define an embedding method that uses the model
def encode_texts_to_embeddings(chunks: List[str]) -> List[Optional[List[float]]]:
    try:
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")

        # convert chunks into list[TextEmbeddingInput]
        inputs = [TextEmbeddingInput(text=chunk, task_type="RETRIEVAL_DOCUMENT") for chunk in chunks]
        embeddings = model.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(chunks))]


# Generator function to yield batches of descriptions
def generate_batches(
    chunks: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def encode_text_to_embedding_batched(
    chunks: List[str], api_calls_per_minute: int = 20, batch_size: int = 5
) -> Tuple[List[bool], np.ndarray]:

    embeddings_list: List[List[float]] = []

    # Prepare the batches using a generator
    batches = generate_batches(chunks, batch_size)

    seconds_per_job = 60 / api_calls_per_minute

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total=math.ceil(len(chunks) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(chunks, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return is_successful, embeddings_list_successful

embeddings = encode_text_to_embedding_batched(pages, api_calls_per_minute=100)
embeddings_array = embeddings[1]
ids = [i for i in range(len(pages))]    

import json

FILE_NAME = "embeddings.json"

# Create a JSON-L file with properties id and embedding from the ids and embeddings array
with open(FILE_NAME, "w") as f:
    for id, embedding in zip(ids, embeddings_array):
        f.write(json.dumps({"id": str(id), "embedding": embedding.tolist()}) + "\n")

# Create a Cloud Storage bucket if it does not already exist
from google.cloud import storage

# Make the bucket name below unique
BUCKET_NAME = f'{BUCKET_NAME}-embedding'
BUCKET_URI = "gs://{0}".format(BUCKET_NAME)

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
if not bucket.exists():
   bucket.create(location="us-central1")

# Upload the JSON-L file to the bucket
blob = bucket.blob(FILE_NAME)
blob.upload_from_filename(FILE_NAME)

# print the URI of the bucket
print(BUCKET_URI)

## Create a Firestore database in the Console, then run the following code
from google.cloud import firestore

db = firestore.Client()

# Create a collection to store the search results
collection_name = "pdf_pages"
if collection_name not in db.collections():
    db.collection(collection_name)

# Add documents to the collection with ids from the ids collection and pages from the pages collection
for id, page in zip(ids, pages):
    db.collection(collection_name).document(str(id)).set({"page": page})

# Initialization
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=LOCATION)

my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name = "assessment-index",
    contents_delta_uri = BUCKET_URI,
    dimensions = 768,
    approximate_neighbors_count = 5,
)

my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name = f"assessment-index-endpoint",
    public_endpoint_enabled = True
)

my_index_endpoint.deploy_index(
    index = my_index, deployed_index_id = "assessment_index_deployed"
)

print("Python Script Ended")