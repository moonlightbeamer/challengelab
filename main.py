import os
from datetime import datetime
import pypdf
import yaml
import json
from flask import Flask, render_template, request

import firebase_admin
from firebase_admin import firestore

from google.cloud import storage

from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai.preview.generative_models as generative_models



PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
SOURCE_FILE_NAME = "fpc-manual.pdf"
JSONL_FILE_NAME = "embeddings.json"
UID = datetime.now().strftime("%m%d%H%M")
BUCKET_NAME = "bkt-{PROJECT_ID}-{UID}"
JSONL_FILE_PATH = f"gs://{BUCKET_NAME}"
EMBEDDING_MODEL_NAME = "textembedding-gecko@002"
INDEX_DIMENSION = 768
INDEX_APPROXIMATE_NEIGHBORS_COUNT = 10
INDEX_NAME = "assessment-index"
DEPLOYED_INDEX_ID = "assessment_index_deployed"
INDEX_ENDPOINT_NAME = "assessment-index-endpoint"

# Helper function that reads from the config file. 
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config fle
TITLE = get_config_value(config, 'app', 'title', 'Ask Google')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Your friendly Bot')
CONTEXT = get_config_value(config, 'palm', 'context',
                           'You are a bot who can answer all sorts of questions')
BOTNAME = get_config_value(config, 'palm', 'botname', 'Google')
TEMPERATURE = get_config_value(config, 'palm', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'palm', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'palm', 'top_p', 0.8)
TOP_K = get_config_value(config, 'palm', 'top_k', 40)

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Instantiating the Firebase client
# firebase_app = firebase_admin.initialize_app()
firestore_db = firestore.client(project=PROJECT_ID)

# Instantiate an embedding model here
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")

# Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-1.5-flash-001")

# Open the PDF file
pdf_file = open(SOURCE_FILE_NAME, 'rb')

# Create a PDF reader object
pdf_reader = pypdf.PdfReader(pdf_file)

# Get the number of pages in the PDF
num_pages = len(pdf_reader.pages)
print(f"Read PDF with number of pages: {num_pages}")

with open(JSONL_FILE_NAME, 'w', encoding='utf-8') as f:
  # Loop through each page
  for page_num in range(num_pages):

    # Get the current page
    page = pdf_reader.pages[page_num]
      
    # Extract the text from the page
    text = page.extract_text()
    
    # Write to Firestore
    doc_ref = firestore_db.collection("page_content").document(str(page_num + 1))
    doc_ref.set({'content': text})
    # print(f"Document: {str(page_num + 1)} created in Firestore database successfully.")

    # Get the embeddings for the text
    raw_embeddings_with_metadata = embedding_model.get_embeddings([text])
    embeddings = [embedding.values for embedding in raw_embeddings_with_metadata][0]

    # construct Panda dataframe
    data = {
      "id": str(page_num + 1),
      "embedding": embeddings
    }

    # Write to jsonl file
    f.write(json.dumps(data) + "\n")

    # print(f"Embedding: {str(page_num + 1)} saved in JSONL file successfully.")

# Close the PDF file
pdf_file.close()

# Create a GCS bucket to store the JSONL file
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# Create the bucket if it doesn't exist
try:
  bucket = client.create_bucket(bucket, location=LOCATION)
  # print(f"Bucket {BUCKET_NAME} created successfully.")
except Exception as e:
  print(f"Error creating bucket: {e}")

# Upload the file to the bucket
blob = bucket.blob(JSONL_FILE_NAME)
blob.upload_from_filename(JSONL_FILE_NAME)
print(f"File {JSONL_FILE_NAME} uploaded to bucket {BUCKET_NAME}.")

# Create a vector index from the JSON-L file
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
  display_name = INDEX_NAME,
  contents_delta_uri = JSONL_FILE_PATH,
  dimensions = INDEX_DIMENSION,
  approximate_neighbors_count = INDEX_APPROXIMATE_NEIGHBORS_COUNT,
)

# Wait for the index creation to complete
index.wait()

# print(f"Vector index created: {index.name}")

# Deploy the vector index as an endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
  display_name = INDEX_ENDPOINT_NAME,
  public_endpoint_enabled = True,
  description="Assessment Index Endpoint",
)

endpoint = endpoint.deploy_index(
  index = index, 
  deployed_index_id = DEPLOYED_INDEX_ID,
)

# print(f"Endpoint deployed: {endpoint.resource_name}")

app = Flask(__name__)

# The Home page route
@app.route("/", methods=['POST', 'GET'])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == 'GET':
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else: 
        question = request.form['input']

        # Get the data to answer the question that 
        # most likely matches the question based on the embeddings
        data = search_vector_database(question)

        # Ask Gemini to answer the question using the data 
        # from the database
        answer = ask_gemini(question, data)
        
    # Display the home page with the required variables set
    model = {"title": TITLE, "subtitle": SUBTITLE,
             "botname": BOTNAME, "message": answer, "input": question}
    return render_template('index.html', model=model)


def search_vector_database(question):

    # 1. Convert the question into an embedding
    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    # 3. Get the IDs for the five embeddings that are returned
    # 4. Get the five documents from Firestore that match the IDs
    # 5. Concatenate the documents into a single string and return it

    data = ""
    return data


def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    response = "Not implemented!"
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
