import os
import yaml
from flask import Flask, render_template, request

import firebase_admin
from firebase_admin import firestore

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.language_models import (
    TextEmbeddingModel,
)

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
GEN_MODEL_ID = "gemini-1.5-flash-001"
EMB_MODEL_ID = "textembedding-gecko@002"
INDEX_ENDPOINT_NAME = "assessment-index-endpoint"
DEPLOYED_INDEX_ID = "assessment_index_deployed"

# Instantiating the Firebase client
firebase_app = firebase_admin.initialize_app()
firestore_db = firestore.client()

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

# Instantiate an embedding model here
embedding_model = TextEmbeddingModel.from_pretrained(EMB_MODEL_ID)

# Instantiate a generative model here
gen_model = GenerativeModel(
    GEN_MODEL_ID,
    system_instruction=[
        CONTEXT,
        "You can only answer questions based on the context and data provided. and when you answer it, indicate which page you are getting tha answer from. If you can't find an answer, do not make up an answer, but instead ask user to rephrase their question within your context.", # If the page number is not found, it means it is not from the data provided, so you shouldn't answer this question.
    ],
)

# Set generative model parameters
generation_config = GenerationConfig(
    temperature = TEMPERATURE,
    top_p = TOP_P,
    candidate_count = 1,
    max_output_tokens = MAX_OUTPUT_TOKENS,
)

# Set generative model safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

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
    # # 1. Convert the question into an embedding
    # raw_embeddings_with_metadata = embedding_model.get_embeddings([question])
    # embedding = [embedding.values for embedding in raw_embeddings_with_metadata][0]

    # # 2. Search the Vector database for the 5 closest embeddings to the user's question
    # search_results = INDEX_ENDPOINT_NAME.find_neighbors(
    #     deployed_index_id = DEPLOYED_INDEX_ID,
    #     queries = [embedding],
    #     num_neighbors = 5
    # )

    # # 3. Get the IDs for the five embeddings that are returned
    # ids = [result.id for result in search_results]
    ids = [1, 2, 3, 4, 5]
    # 4. Get the five documents from Firestore that match the IDs
    docs = firestore_db.collection("page_content").where(u'id', 'in', ids).stream()

    # 5. Concatenate the documents into a single string and return it
    data = ""
    for doc in docs:
        doc_data = doc.to_dict()
        doc_text = doc_data.get('content', '')  # the text content is stored in the 'content' field
        data += doc_text + ' '  # Append the document text to data

    # Remove the trailing space
    data = data.strip()
    return data

def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    # SYSTEM_PROMPT = "{CONTEXT} and you can only answer questions based on the data provided below, if you can't find answer, do not hallucinate, just say you can't find answer."
    # Instantiate a Generative AI model here
    prompt = "data: " + data + "\n\n User: " + question + "\n\n Answer: "   # "data: " + data + 
    contents = [prompt]
    try:
        response = gen_model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        response = response.text
    except:
        response = "the code has error."
    return response
        


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
