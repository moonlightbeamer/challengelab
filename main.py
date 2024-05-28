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
    Part,
)
from vertexai.language_models import (
    TextEmbeddingInput, 
    TextEmbeddingModel,
)

import pypdf
import json
from google.cloud import aiplatform
from google.cloud import storage

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
GEN_MODEL_ID = "gemini-1.5-flash-001"
EMB_MODEL_ID = "textembedding-gecko@002"

# Instantiating the Firebase client
firebase_app = firebase_admin.initialize_app()
db = firestore.client()

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
        "You can only answer questions based on the context and data provided. If you can't find an answer, do not make up an answer, but instead ask user to rephrase their question within your context.",
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

# RAG variables definition
SOURCE_FILE_NAME = "fpc-manual.pdf"
JSONL_FILE_NAME = "embeddings.json"
BUCKET_NAME = "{PROJECT_ID}-challengelab"
JSONL_FILE_PATH = f"gs://{BUCKET_NAME}"
INDEX_DIMENSION = 768
INDEX_APPROXIMATE_NEIGHBORS_COUNT = 5
INDEX_NAME = "assessment-index"
DEPLOYED_INDEX_ID = "assessment_index_deployed"
INDEX_ENDPOINT_NAME = "assessment-index-endpoint"
# INDEX_MACHINE_TYPE = "n1-standard-2"
# generate an unique id for this session
from datetime import datetime
UID = datetime.now().strftime("%m%d%H%M")

# Open the PDF file
pdf_file = open(SOURCE_FILE_NAME, 'rb')

# Create a PDF reader object
pdf_reader = pypdf.PdfReader(pdf_file)

# Get the number of pages in the PDF
num_pages = len(pdf_reader.pages)
# print(f"Read PDF with number of pages: {num_pages}")

with open(JSONL_FILE_NAME, 'w', encoding='utf-8') as f:
  # Loop through each page
  for page_num in range(num_pages):

    # Get the current page
    page = pdf_reader.pages[page_num]
      
    # Extract the text from the page
    text = page.extract_text()
    
    # Write to Firestore
    doc_ref = db.collection("page_content").document(str(page_num + 1))
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
  print(f"Bucket {BUCKET_NAME} created successfully.")
except Exception as e:
  print(f"Error creating bucket: {e}")

# Upload the file to the bucket
blob = bucket.blob(JSONL_FILE_NAME)
blob.upload_from_filename(JSONL_FILE_NAME)
# print(f"File {JSONL_FILE_NAME} uploaded to bucket {BUCKET_NAME}.")

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
#     # 1. Convert the question into an embedding
#     raw_embeddings_with_metadata = embedding_model.get_embeddings([question])
#     embedding = [embedding.values for embedding in raw_embeddings_with_metadata][0]

#     # 2. Search the Vector database for the 5 closest embeddings to the user's question
#     search_results = vertexai.search_vectors(embedding, 5)
#     search_vector_index(
#     query_text=prompt_text,
#     index_id="your-index-id",
#     top_k=5,
# )

#     # 3. Get the IDs for the five embeddings that are returned
#     ids = [result.id for result in search_results]

#     # 4. Get the five documents from Firestore that match the IDs
#     docs = db.collection("page_content").where(u'id', 'in', ids).stream()

#     # 5. Concatenate the documents into a single string and return it
#     data = ""
#     for doc in docs:
#         doc_data = doc.to_dict()
#         doc_text = doc_data.get('content', '')  # the text content is stored in the 'content' field
#         data += doc_text + ' '  # Append the document text to data

#     # Remove the trailing space
#     data = data.strip()
#     return data

    data = """fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
INTRODUCTION
The New York City Department of Health and Mental
Hygiene has the jurisdiction to regulate all matters
affecting health in the city and to perform all those
functions and operations that relate to the health of the people
of the city.
INTRODUCTION
The United States has one of the
safest food safety systems in the world,
yet millions of Americans still get
sick each year from eating contaminated
foods; hundreds of thousands
are hospitalized; and several thousand
die. This means that there is
still tremendous room for improvement
in food safety standards.
Most food-borne illnesses are
caused by improper handling of
food. The statistics from the Centers
for Disease Control (CDC) show
that some of the most common
causes of foodborne illness are:
  Sick food worker
  Poor personal Hygiene/Bare hand
contact
  Improper holding temperatures
  Improper cooling
  Inadequate cooking and reheating
  Cross contamination
  Use of food from unknown source
What is Food-Borne Illness?
Any illness that is caused by food
is called food-borne illness. A foodborne
illness outbreak is defined as
any incident involving two or more
persons becoming ill with similar
symptoms from the same source.
Typically these illnesses are a direct
result of contamination of food by
harmful microorganisms, (commonly
called germs) such as bacteria,
viruses, parasites, fungi etc. Injury
and illness caused by foreign objects,
dangerous chemicals and/or allergens
in food is also considered a foodborne
illness.
Who is at Risk?
We are all at risk of getting a food
borne illness; however, the effects are
more severe for certain categories of
individuals:
  Children whose immune system
(human body’s defense system
against diseases) is not fully developed
yet.
  Elderly individuals because their
immune system is not robust anymore
and has weakened due to old
age.
  Pregnant women where the threat
is both to the mother and the fetus.
  Individuals with compromised
immune systems
e.g., Patients with
AIDS, cancer or individuals
who are diabetics,
etc.
  People on medication
(antibiotics, immunosuppressant,
etc.).
What is food?
Food is any edible substance,
ice, beverage, or
ingredient intended for use
and used or sold for
human consumption.
What are Potentially Hazardous
Foods (PHF)?
This expression refers to those foods
that provide suitable conditions for
rapid growth of microorganisms.
These include foods that are high in
protein like raw or cooked animal
products such as meats, poultry,
fish, shellfish (mollusks as well as
crustaceans), milk and milk products
(cheese, butter milk, heavy cream etc.,),
plant protein such as tofu, and
starches such as cooked rice, cooked
pasta, cooked beans and cooked
vegetables like potatoes, cut melons,
cut leafy greens, cut tomatoes or
mixtures of cut tomatoes, as well as
raw seed sprouts and garlic in oil.
Exceptions: Those foods that have a
low water activity (.85 or less) or those
that are highly acidic with a pH of
4.6 or below. Air-cooled hard-boiled
eggs with shells intact.
2
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
INTRODUCTION TO FOOD SAFETY
QUICK REVIEW
1. All food service establishments must have a current and valid permit issued by the NYC Health Department.
  TRUE   FALSE
2. Health Inspectors have the right to inspect a food service or food processing establishment as long as it is in operation.
Inspectors must be given access to all areas of establishment during an inspection.   TRUE   FALSE
3. Health Inspectors are authorized to collect permit fees and fines on behalf of the Department.   TRUE   FALSE
4. Health Inspectors must show their photo identification and badge to the person in charge of an establishment.
  TRUE   FALSE
5. According to the NYC Health Code, who is required to have a Food Protection Certificate? ________________________________.
Potentially Hazardous Foods
What is Ready-To- Eat Food?
Any food product that does not
need additional heat treatment or
washing is called ready-to-eat food.
Extra care must be taken to ensure
the safety of these foods.
Where do we purchase foods?
All foods must be purchased from
approved sources. These are manufacturers
and suppliers who comply
with all the rules and regulations that
pertain to the production of their
product, including having the
How do we store potentially
hazardous foods?
All foods must be kept free from
adulteration, spoilage, filth or
other contamination in order to be
suitable for human consumption.
Potentially hazardous foods are of
particular concern because they
provide the conditions suitable for
the growth of microorganisms.
These foods must be kept either hot or
cold to prevent microorganisms from
growing. Hot means 140°F or above
and cold means 41°F or below. The
temperature range between 41°F and
140°F is known as the temperature
danger zone. It is within this range
that microorganisms are comfortable
and will grow rapidly. At 41°F and
below, the temperature is cold
enough to retard or slow down the
growth of microorganisms, while
above 140°F most of the microorganisms
which cause foodborne illness
begin to die.
Thermometers
The only safe way to determine
that potentially hazardous foods are
kept out of the temperature danger
zone is by the use of thermometers.
There are several different types of
thermometers. The bi-metallic stem is
the most popular type. It is fairly
inexpensive, easy to use, accurate to
+ or – 2°F and easy to re-calibrate.
Also, it is available within the range
of 0° to 220°F making it ideal for
measuring the required temperatures
in a food establishment.
Another thermometer in use is
the thermocouple which is very accurate
but fairly expensive. Lastly, there is
a thermometer called thermistor,
which has a digital read out and is
commonly called "digital thermometer."
These thermometers are used by
inserting the probe into the thickest
part or the geometric center of the
food item being measured. The stem
thermometer must remain in the food
until the indicator stops moving before
the reading is taken and must be recalibrated
periodically to assure accuracy.
Calibration
Thermometers must be calibrated
to ensure their accuracy. For thermocouple
thermometers, follow the
instructions provided by the manufacturer.
For some thermistor thermometers,
placing the thermometer
in 50/50 solution of ice and water
or boiling water, and hitting the
“reset” button will automatically
calibrate the thermometer. Bi-metallic
stem thermometers may be calibrated
by two methods:
Boiling-Point Method
Ice-Point method
Boiling-Point Method
  Bring water to a boil.
  Place the thermometer probe (stem)
into the boiling water. Make sure
that the thermometer probe does
not touch the bottom or sides of
the pan. Wait until the indicator
stops moving, then record the
temperature.
  If the temperature is 212°F, do
nothing, the thermometer is accurate.
(This is the temperature of boiling
water at sea level.)
  If the temperature is not 212°F,
rotate the hex-adjusting nut using
a wrench or other tool until the
indicator is at 212°F.
3
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Bi-metallic
Thermometer
Cold temperature
reading calibration
Thermocouple
Thermometer
necessary permits to operate. The use of
foods prepared at home or in an unlicensed
establishment is prohibited.
The Temperature Danger Zone?
Most microorganisms that cause
foodborne illness typically grow best
between temperatures of 41°F and
140°F. This is commonly referred to
as the temperature danger zone. One
of the basic and simplest ways to keep
food safe is by keeping it out of the
temperature danger zone.
212°
165°
140°
41°
32°
0°
DANGER�
ZONE
Temperature
Danger
Zone
Ice-Point Method
  Fill a container with ice and water to
make a 50/50 ice water slush.
  Stir the slush.
  Place the thermometer probe so
that it is completely submerged in
the ice-water slush, taking care
not to touch the sides or the bottom.
Wait until the indicator needle
stops moving, then record the
temperature.
  If the temperature is 32°F, do
nothing, the thermometer is
accurate. (A 50/50 ice water slush
will always have a temperature of
32°F at sea-level.) If the tempearture
is not 32°F, rotate the hexadjusting
nut until the indicator
needle is at 32°F.
How to use a Thermometer
The following describes the proper
method of using thermometers:
  Sanitize the probe by the use of
alcohol wipes. This is a fairly safe
and common practice. Other
methods such as immersion in
water with a temperature of 170°F
for 30 seconds or in a chemical
sanitizing solution of 50 PPM for
at least one minute, or swabbing
with a chlorine sanitizing solution
of 100 PPM are also acceptable.
  Measure the internal product
temperature by inserting the probe
into the thickest part or the center
of the product. It is recommended
that the temperature readings
be taken at several points.
  Whenever using a bi-metallic
thermometer, ensure that the
entire sensing portion – from the
tip of the probe to the indentation
on the stem, is inserted in to
the food product.
  Wait for roughly 15 seconds or
until the reading is steady before
recording it.
  Clean and sanitize the thermometer
for later use.
The first opportunity one has to
ensure that food is safe is at the
point of receiving. At this point care
must be taken to ensure that all
products come from approved
sources and/or reliable and reputable
suppliers. Incoming supplies
must be received at a time when it
is convenient to inspect them and
place them into storage promptly.
There are various qualities and conditions
one should look for in different
food items.
Beef
Incoming supplies of beef can be
received either fresh or frozen. Fresh
beef should be at 41°F or below
while frozen beef should be at 0°F
or below. Beef should be bright to
dark red in color with no objectionable
odor. To ensure that the supply
is from an approved source, look for
the United States Department of
Agriculture inspection stamp. This
can be found on the sides of the beef
carcass or on the box when receiving
portions of the carcass. This inspection
is mandatory and the stamp indicates
that the meat is sanitary, wholesome
and fit for human consumption. Also
found may be a grade stamp which
attests to the quality of the meat and
will certainly have an impact on its
price. The inspection stamp is the
more important of the two stamps.
Lamb
Lamb, like beef, may have an
inspection stamp as well as a grade
stamp. When fresh, it is light red in
color and has no objectionable odor
and the flesh is firm and elastic. Fresh
lamb is received at 41°F and frozen
at or below 0°F. (See stamps below)
Pork
Pork is also subject to USDA
inspection. The flesh is light colored
while the fat is white. A good way
to check for spoilage is to insert a
knife into the flesh all the way to
the bone and check the blade for
any off odors. (See stamps below)
Chicken and Poultry
Chicken and poultry are subject
to USDA inspection which must be
verified by the inspection stamp.
(See stamps below) These must be
received either fresh at 41°F and
below or frozen at 0°F or less, as
they are naturally contaminated
with the micro-organism Salmonella
which must be kept under control.
4
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
USDA Poultry
Inspection Stamp
USDA Poultry
Grade Stamp
USDA Meat
Inspection Stamp
USDA Meat
Grade Stamp
RECEIVING FOODS
Fresh fish
There is no inspection for fresh
fish other than what can be done by
sight and touch and one’s sense of
smell. This makes it more important
to purchase supplies from reputable
and reliable suppliers. Fresh
fish must be received cold and on
ice, 41°F or less, with no objectionable
odor. The eyes must be clear
and bulging, the gills bright red and
the flesh firm and elastic. Fish that
is spoiling will have a fishy odor;
the eyes cloudy, red rimmed and
sunken; the gills grey or greenish;
the flesh will pit on pressure and
can easily be pulled away from the
bones; the scales are loose.
Smoked fish
Smoked fish provide ideal conditions
for the growth of Clostridium
botulinum spores if left at room
temperature. Therefore, upon receipt,
all smoked fish must be stored at
38°F or below.
It is important to adhere to the
temperature requirements stated on
the label.
Shellfish
Shellfish is the term used to
describe clams, mussels, and oysters.
These belong to the family of mollusks.
They are filter feeders, that is,
they absorb water from their environment,
filter out whatever nutrients
are there and then expel the
water. Feeding in this manner causes
them to absorb and accumulate
harmful microorganisms from polluted
waters. Since the whole shellfish
is eaten either raw or partially
cooked, it is critical to ensure that
they are harvested from safe waters.
It is important to buy shellfish from
reputable suppliers who can provide
the shipper’s tags which identify the
source of the shellfish. These tags
supply the following information:
  The name of the product
  The name of the original shipper
  The address of the original shipper
  The interstate certificate number
of the original shipper
  The location of the shellfish harvesting
area.
When purchasing small amounts
from a retailer, a tag must be provided.
This is a split-lot tag which
has all the information that is on
the original tag.
The shellfish tag is required to be
kept together with the product, then
whenever the product is used up, it
must be kept for 90 days in order of
delivery. The virus Hepatitis A is
associated with shellfish.
Check if the shellfish is alive. An
opened shell may be an indication
of dead shellfish. Gently tap on the
shell, if the shell closes then it is
alive otherwise it’s dead and should
be discarded. Both alive as well as
shucked shellfish (shellfish that has
been removed from its shell) must
only be accepted if delivered at a
temperature of 41°F or below.
Following conditions would automatically
be grounds for rejection:
  Slimy, sticky or dry texture
  Strong fishy odor
  Broken shells
Other Shellfish
Lobsters, crabs and shrimps belong
to the family of crustaceans. Fresh
lobsters and crabs must be alive at
the time of delivery. As with other
seafood, a strong fishy odor is an
indication of spoilage. The shell of
the shrimp must be intact and firmly
attached. All processed crustacean
must be delivered at 41°F or below.
Eggs
Eggs produced outside of New
York State are inspected by the U.S.
Department of Agriculture while
those produced within the State are
inspected by the New York State
Department of Agriculture and
Markets. In either case, inspected
eggs will be identified by a stamp
on the carton. Eggs have long been
5
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Split Lot Tag
Shellfish Tag
It is strongly recommended that the invoices be kept with
the tags to aid in tracing the lot’s history.
associated with the micro-organism
Salmonella enteritidis. This bacterium
has been found on the inside of
the egg, so external washing does
not make eggs safe.
Eggs should be bought from suppliers
who deliver them in refrigerated
trucks and upon receipt, these
eggs must be kept refrigerated at an
ambient temperature of 45°F until
they are used.
Pasteurized Eggs
Pasteurization is a method of
heating foods to destroy harmful
microorganisms. Pasteurized eggs
come in many forms: intact shell eggs,
liquid eggs, frozen eggs, or in powdered
form. Even though these have
been pasteurized, they still require
refrigeration to slow down growth
of spoilage microorganisms to extend
the shelf life. Only the powdered
pasteurized eggs may be held at room
temperature.
Milk and Milk Products
Only accept Grade A pasteurized
milk and milk products. Harmful
pathogens such as Listeria monocytogenes,
E.coli 0157:H7 and
Salmonella spp. are commonly associated
with un-pasteurized milk.
The expiration date on pasteurized
milk and milk products must
not exceed nine calendar days from
date of pasteurization, while ultra
pasteurized milk and milk products
must not exceed 45 days from date
of ultra pasteurization.
Upon receipt, these products
must be checked to ensure that they
are well within the expiration period
and that they are at 41°F or below.
This temperature must be maintained
until the product is used up.
Fresh Fruits and Vegetables
The acceptable condition of fruits
and vegetables vary from one item
to another. As a general rule of thumb,
only accept those that do not show
any signs of spoilage. Reject any
produce that shows signs of decay,
mold, mushiness, discoloration,
wilting, and bad odors.
A recent study done by the center
for Science in the Public Interest
(CSPI) found that contaminated fruits
and vegetables are causing more foodborne
illness among Americans than
raw chicken and eggs combined.
Most fresh produce may become
contaminated with Salmonella and
E.coli 0157:H7 due to the
use of manure fertilizer
(more common in South
and Central America, which
is a major source of fresh
produce to the United
States).
Fresh produce must be
thoroughly washed prior to
being served raw. This
includes all kinds of fruits
and vegetables including
produce that has a hard
rind that is typically not
consumed, for example,
watermelons, cantaloupes, honey
dews and all varieties of melons,
oranges, etc. Only potable running
water should be used to thoroughly
wash these produce, and the use of
produce scrubbing brushes is
strongly recommended.
Canned Goods
It is a simple task to inspect
canned goods and remove from circulation
those cans that can cause
foodborne illness. The first step is
to ensure that home canned foods
are not used in a food service establishment.
All canned foods must be
commercially processed. A good can
is free from rust and dents, properly
sealed and labeled and slightly concave
at both ends.
A can with a dent on any of the
three seams (top, bottom or side)
must be removed from circulation.
The same requirement is true for
severely rusted, severely dented, leaking
and cans with swollen ends. Bad
cans may be rejected at delivery or segregated
and clearly labeled for return
to the supplier.
6
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
swollen severe dents slight rust
critical major minor
Egg Cartons Stamps
Modified Atmosphere
Packaged Foods
Various food items are packaged
under special conditions to prolong
their shelf life. These conditions
include the following:
  Food is placed in a package and
all the air is withdrawn: vacuum
packaging.
  Food is placed in a package, all
the air is withdrawn and gases are
added to preserve the contents –
modified atmosphere packaging.
  Food is placed in a package, all
the air is withdrawn and the food
is cooked in the package: sous vide
packaging.
Because of the absence of air, foods
packaged in this manner provide ideal
conditions for the growth of the
clostridium botulinum micro-organism,
unless they are refrigerated at temperatures
recommended by the
manufacturer.
These products must be provided
by approved sources and care taken
to preserve the packaging during
handling and when taking the temperature.
Food establishments interested in
making “modified atmosphere packaged
foods” must first obtain permission
from NYC DOHMH.
For more information , please see
Page 54.
Dry Foods
Dry foods such as grains, peas,
beans, flour and sugar are to be dry
at the time of receiving. Moisture
will cause growth of molds and the
deterioration of these products.
Broken and defective packages will
indicate contamination; as will the
evidence of rodent teeth marks.
Whenever these products are
removed from their original containers,
they must be stored in
tightly covered, rodent-proof
containers with proper labels.
7
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Refrigerated and Frozen
Processed Foods
For convenience as well as cutting
down on costs, there has been a greater
shift towards using prepared prepackaged
refrigerated or frozen foods.
These routinely include deli and
luncheon meats, refrigerated or frozen
entrees, etc. Care should be taken
when receiving these products to ensure
quality as well as safety. Following
are some guidelines:
  Ensure that refrigerated foods are
delivered at 41°F or below.
(Except, as noted previously,
smoked fish must be received at
38° F or lower.)
  Ensure that frozen foods are
delivered at 0°F or lower.
  All packaging must be intact.
  Any frozen food packaging that
shows signs of thawing and
refreezing should be rejected.
Signs include liquid or frozen liquids
on the outside packaging, formation
of ice crystals on the packaging or
on the product, and water stains.
QUICK REVIEW
1. The term "potentially hazardous food" refers to foods which do not support rapid
growth of microorganisms.   TRUE   FALSE
2. Home canned food products are allowed in commercial food
establishments.   TRUE   FALSE
3. The Temperature Danger Zone is between 41°F and 140°F.
  TRUE   FALSE
4. Within the Temperature Danger Zone, most harmful microorganisms
reproduce rapidly.   TRUE   FALSE
5. Shellfish tags must be filed in order of delivery date and kept for a period
of _______ days.
6. Fresh shell eggs must be refrigerated at an ambient temperature
of: ______°F.
7. Foods in Modified Atmosphere Packages provide ideal conditions for the growth
of: _______
8. The recommended range of bi-metallic stem thermometer is: _______
9. Meat inspected by the U.S. Dept. of Agriculture must have a/an:
____________ stamp.
10. Chicken and other poultry are most likely to be contaminated with: _______
11. Smoked fish provide ideal conditions for the growth of Botulinum spores.
Therefore, this product must be stored at: ______°F
12. Safe temperatures for holding potentially hazardous foods are: ______°F or
below and ______°F or above
13. What are the four types of defective canned products that must be
removed from circulation? ______, ______, _____, _____
14. Which of the following is an indication that fish is not fresh?:
  clear eyes   fishy odor   firm flesh
After receiving the foods proper
ly, they must be immediately
moved to appropriate storage areas.
The most common types of food
storage include:
Refrigeration storage
Freezer storage
Dry storage
Storage in Ice
We will discuss each of these
individually; however, certain
aspects are common for all types of
storage and are described below.
FIFO
An important aspect of food storage
is to be able to use food products
before their “use-by” or expiration
date. In this regard, stock rotation is
very important. The common sense
approach of First in First out (FIFO)
method of stock rotation prevents
waste of food products and ensures
quality. The first step in implementing
the FIFO method of stock rotation
is to date products. Marking the
products with a date allows food
workers to know which product was
received first. This way, the older stock
is moved to the front, and the newly
received stock is placed in the back.
Storage Containers
It is always best to store food in
their original packaging; however,
when it is removed to another container,
take extra care to avoid contamination.
Only use food containers
that are clean, non-absorbent and
are made from food-grade material
intended for such use. Containers
made from metal may react with
certain type of high acid foods such
as sauerkraut, citrus juices, tomato
sauce, etc. Plastic food-grade containers
are the best choice for these
types of foods. Containers made of
copper, brass, tin and galvanized metal
should not be used. The use of such
products is prohibited.
Re-using cardboard containers to
store cooked foods is also a source
of contamination. Lining containers
with newspapers, menus or other
publication before placing foods is
also prohibited as chemical dyes from
these can easily leach into foods.
Storage Areas
Foods should only be stored in
designated areas. Storing foods in
passageways, rest rooms, garbage
areas, utility rooms, etc. would subject
these to contamination. Raw
foods must always be stored
below and away from cooked
foods to avoid cross contamination.
Refrigerated Storage
This type of storage is typically
used for holding potentially
hazardous foods as well as
perishable foods for short periods
of time—a few hours to a
few days.
An adequate number of efficient
refrigerated units are
required to store potentially
hazardous cold foods. By keeping
cold foods cold, the microorganisms
that are found naturally on these
foods are kept to a minimum. Cold
temperature does not kill microorganisms,
however, it slows down
their growth.
Pre-packaged cold foods must be
stored at temperatures recommended
by the manufacturer. This is especially
important when dealing with vacuum
packed foods, modified atmosphere
packages and sous vide foods. Smoked
fish is required by the Health Code to
be stored at 38°F or below.
Fresh meat, poultry and other
potentially hazardous foods must be
stored at 41°F or below, while frozen
foods must be stored at 0°F or below.
For foods to be maintained at these
temperatures, refrigerators and
freezers must be operating at temperatures
lower than 41°F and 0°F.,
respectively. Thermometers placed
in the warmest part of a refrigerated
unit are necessary to monitor the
temperature of each unit.
The rule of storage, First In First
Out (FIFO) ensures that older
deliveries are used up before newer
ones. In practicing FIFO, the very
first step would be to date all products
as they are received. The next
step is to store the newer products
behind the older ones.
The following rules are important
in making sure that foods are safe
during refrigerated storage:
  Store cooked foods above raw
foods to avoid cross-contamination.
  Keep cooked food items covered
unless they are in the process of
cooling, in which case they must be
covered after being cooled to 41°F.
  Avoid placing large pots of hot
foods in a refrigerator. This will
cause the temperature of the
refrigerator to rise and other
foods will be out of temperature.
8
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
STORAGE OF FOOD
Cross Contamination
When harmful microorganisms are
transferred from one food item to
another, typically, from raw foods to
cooked or ready to eat foods, it is
termed cross contamination. This
expression also applies in any situation
where contamination from one
object crosses over to another. Cross
contamination may also occur between
two raw products, for instance,
poultry juices falling on raw beef
will contaminate it with Salmonella,
which is typically only associated
with poultry and raw eggs.
  Store foods away from dripping
condensate , at least six inches above
the floor and with enough space
between items to encourage air
circulation.
Freezer Storage
Freezing is an excellent method
for prolonging the shelf life of foods.
By keeping foods frozen solid, the
bacterial growth is minimal at best.
However, if frozen foods are thawed
and then refrozen, then harmful
bacteria can reproduce to dangerous
levels when thawed for the second
time. In addition to that, the quality of
the food is also affected. Never refreeze
thawed foods, instead use them
immediately. Keep the following
rules in mind for freezer storage:
  Use First In First Out method of
stock rotation.
  All frozen foods should be frozen
solid with temperature at 0°F or
lower.
  Always use clean containers that
are clearly labeled and marked,
and have proper and secure lids.
  Allow adequate spacing between
food containers to allow for proper
air circulation.
  Never use the freezer for cooling
hot foods.
* Tip: When receiving multiple
items, always store the frozen foods
first, then foods that are to be refrigerated,
and finally the non perishable
dry goods.
Dry Storage
Proper storage of dry foods such
as cereals, flour, rice, starches, spices,
canned goods, packaged foods and
vegetables that do not require refrigeration
ensures that these foods will
still be usable when needed. Adequate
storage space as well as low humidity
(50% or less), and low temperatures
(70 °F or less) are strongly recommended.
In addition to the above,
avoid sunlight as it may affect the
quality of some foods. Following are
some of the guidelines:
  Use First In First Out method of
stock rotation.
  Keep foods at least 6 inches off the
floor. This allows for proper cleaning
and to detect vermin activity.
  Keep foods in containers with
tightly fitted lids.
  Keep dry storage areas well lighted
and ventilated.
  Install shades on windows to prevent
exposure from sunlight.
  Do not store foods under overhead
water lines that may drip
due to leaks or condensation.
  Do not store garbage in dry food
storage areas.
  Make sure that dry storage area is
vermin proof by sealing walls and
baseboards and by repairing holes
and other openings.
* Safety Tip: Storage of harmful
chemicals in the food storage areas
can create hazardous situations and
hence is prohibited by law. All chemicals
must be labeled properly and
used in accordance to the instructions
on the label. Pesticide use is prohibited
unless used by a licensed pest control
officer.
Storage in Ice
Whenever food items are to be
stored in ice, care must be taken to
ensure that water from the melted
ice is constantly being drained so
that the food remains on ice and
not immersed in iced water.
Furthermore, it is improper to
store food in ice machines or ice
that will be later used for human
consumption.
  Food should be stored at least six
inches off the floor, away from walls
and dripping pipes.
  Keep all food, bulk or otherwise,
covered and safe from contamination.
  Check food daily and throw away
any spoiled or contaminated food.
  Store cleaning, disinfecting, and
other chemicals away from foods,
clearly marked and in their original
containers.
  Keep food refrigerated at a temperature
of 41°F or below.
  Monitor temperatures regularly
with a thermometer placed in the
warmest part of the refrigerator.
  Keep all cooling compartments
closed except when you are using
them.
  Store food in a refrigerator in
such a way that the air inside can
circulate freely.
  Keep all refrigerated foods covered,
and use up stored leftovers quickly.
  When dishes and utensils are
sparkling clean, keep them that
way by proper storage. Keep all
cups and glasses inverted.
  Cakes, doughnuts and fruit pies
may be kept inside a covered display
area.
  The only goods that should be left
on the counter uncovered are those
which are individually wrapped
and not potentially hazardous.
  Do not set dirty dishes, pots, cartons
or boxes on food tables.
  Whenever products are removed
from their original containers, store
them in tightly covered, rodent
proof containers with labels.
9
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
FOOD STORAGE REVIEW
Foods stored at least six
inches above the floor
10
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Product Refrigerator Freezer
Eggs
Fresh, in shell 4 to 5 weeks Don’t freeze
Raw yolks, whites 2 to 4 days 1 year
Hard cooked 1 week Don’t freeze well
Liquid pasteurized eggs
or egg substitutes,
opened 3 days Don’t freeze
unopened 10 days 1 year
Mayonnaise, commercial
Refrigerate after opening 2 months Don’t freeze
TV Dinners, Frozen Casseroles
Keep frozen until ready to heat 3 to 4 months
Deli & Vacuum-Packed Products
Store-prepared 3 to 5 days Don’t freeze well
egg, chicken, tuna, ham,
macaroni salads
Pre-stuffed pork & lamb
chops, chicken breasts
stuffed w/dressing 1 day Don’t freeze well
Store-cooked convenience
meals 3 to 4 days Don’t freeze well
Commercial brand
vacuum-packed
dinners with USDA seal,
unopened 2 weeks Don’t freeze well
Raw Hamburger, Ground & Stew Meat
Hamburger & stew meats 1 to 2 days 3 to 4 months
Ground turkey, veal, pork,
lamb 1 to 2 days 3 to 4 months
Ham, Corned Beef
Corned beef in pouch
with pickling juices 5 to 7 days Drained, 1 month
Ham, canned, labeled
“Keep Refrigerated,”
unopened 6 to 9 months Don’t freeze
opened 3 to 5 days 1 to 2 months
Ham, fully cooked, whole 7 days 1 to 2 months
Ham, fully cooked, half 3 to 5 days 1 to 2 months
Ham, fully cooked, slices 3 to 4 days 1 to 2 months
Hot Dogs & Lunch Meats (in freezer wrap)
Hot dogs,
opened package 1 week 1 to 2 months
unopened package 2 weeks 1 to 2 months
Lunch meats,
opened package 3 to 5 days 1 to 2 months
unopened package 2 weeks 1 to 2 months
Refrigerator and Freezer Storage Chart
Since product dates aren’t a guide for safe use of a product, consult this chart and follow these tips.
These short but safe time limits will help keep refrigerated food 41° F (5°C) from spoiling or becoming dangerous.
• Purchase the product before “sell-by” or expiration dates.
• Follow handling recommendations on product.
• Keep meat and poultry in its package until just before using.
• If freezing meat and poultry in its original package longer than 2 months, overwrap these packages
with airtight heavy-duty foil, plastic wrap, or freezer paper, or place the package inside a plastic bag.
Because freezing 0° F (-18° C) keeps food safe indefinitely, the following recommended storage times are
for quality only.
RECOMMENDED STORAGE OF FOOD
Product Refrigerator Freezer
Soups & Stews
Vegetable or meat-added
& mixtures of them 3 to 4 days 2 to 3 months
Bacon & Sausage
Bacon 7 days 1 month
Sausage, raw from pork,
beef, chicken or turkey 1 to 2 days 1 to 2 months
Smoked breakfast links,
patties 7 days 1 to 2 months
Summer sausage labeled
“Keep Refrigerated,”
unopened 3 months 1 to 2 months
opened 3 weeks 1 to 2 months
Fresh Meat (Beef, Veal, Lamb, & Pork)
Steaks 3 to 5 days 6 to 12 months
Chops 3 to 5 days 4 to 6 months
Roasts 3 to 5 days 4 to 12 months
Variety meats (tongue,
kidneys, liver, heart,
chitterlings) 1 to 2 days 3 to 4 months
Meat Leftovers
Cooked meat & meat dishes 3 to 4 days 2 to 3 months
Gravy & meat broth 1 to 2 days 2 to 3 months
Fresh Poultry
Chicken or turkey, whole 1 to 2 days 1 year
Chicken or turkey, parts 1 to 2 days 9 months
Giblets 1 to 2 days 3 to 4 months
Cooked Poultry, Leftover
Fried chicken 3 to 4 days 4 months
Cooked poultry dishes 3 to 4 days 4 to 6 months
Pieces, plain 3 to 4 days 4 months
Pieces covered with broth,
gravy 1 to 2 days 6 months
Chicken nuggets, patties 1 to 2 days 1 to 3 months
Fish & Shellfish
Lean fish 1 to 2 days 6 months
Fatty fish 1 to 2 days 2 to 3 months
Cooked fish 3 to 4 days 4 to 6 months
Smoked fish 14 days 2 months
Fresh shrimp, scallops,
crawfish, squid 1 to 2 days 3 to 6 months
Canned seafood after opening out of can
Pantry, 5 years 3 to 4 days 2 months
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E 11
Food borne illnesses are caused
by the presence of foreign
objects, chemicals and living organisms
in our foods. These can be
described as hazards to our health.
Physical Hazards
The presence of a foreign object
in food that can cause an injury or
an illness is called a Physical Hazard.
The common cause of a physical
hazard is accidental and/or due to
improper food handling practices by
food workers. Food workers must
be trained to handle foods safely so
as not to contaminate foods. Food
workers should not wear jewelry or
any other personal effects that may
accidentally fall into food items.
Some common examples include:
  Tiny pebbles that are sometimes
found in rice, beans, and peas.
  Fragments of glass—from a broken
glass, from scooping ice with
the glass, from broken light bulb
without protective shields, etc.
  Short, un-frilled toothpicks used
to hold a sandwich together.
  Bandages
  Metal shavings from a worn can
opener
  Scouring pad (steel wool) wire
  Pieces of jewelry
Any food item with a physical hazard
must be discarded immediately.
Chemical hazards
A chemical hazard may be in a
food item either accidentally, deliberately
or naturally.
A chemical may be introduced to
a food accidentally by the careless
use of insecticides, storing of cleaning
and other chemicals next to
open foods and the storage of acidic
foods in metal containers.
These are the more common
examples and may be avoided by:
  Using an experienced, licenced
exterminator.
  Storing cleaning and other chemicals,
including personal medication,
away from foods, preferably
in a locked cabinet.
  Storing acidic foods in containers
made of food-grade plastic.
A chemical may be introduced
into a food item deliberately to
enhance its taste or appearance
without realizing that it may cause
consumers to become ill.
Sulfites are used to maintain the
color and freshness of cut fruits and
vegetables.
Monosodium Glutamate (MSG)
is used to enhance the flavor of foods.
Excessive use of sulfites and MSG
have both resulted in serious allergic
reaction among sensitive individuals.
MSG is permitted in a food service
establishment as long as it is disclosed
on the menu, however, the
use of sulfites is prohibited. Certain
foods may contain sulfites when they
are brought in but none may be added
in a food service establishment.
Toxic metals
Utensils made from lead, copper,
brass, zinc, antimony and cadmium
are not permitted for use with food
products. These can cause toxicmetal
poisoning from the leaching
of these chemicals into the food.
Similarly, containers previously
designed to hold cleaning agents
and other chemicals should never be
used for food storage. Always ensure
that food storage containers are
made from food-grade materials.
Biological Hazards
Biological hazards occur when
disease-causing microorganisms
such as Bacteria, Viruses, Parasites
and Fungi end up contaminating
our food supply. In addition to that,
toxins found naturally in certain foods
can also cause food borne illness.
Mushrooms are both poisonous
and non-poisonous and they are difficult
to tell apart. To be certain
that a safe variety is being used,
they must be purchased from a reliable
commercial source.
Toxins in certain fish can also be
a serious problem. Some fish have
natural toxins, others accumulate
toxins from their food, while yet
QUICK REVIEW
1. The acronym FIFO means: ___________________
2. The first step in implementing FIFO is to rotate the stock.
  TRUE   FALSE
3. The New York City Health code requires that all food items must be
stored at least _______ off the floor.
4. In order to prevent cross-contamination, raw foods in a refrigerator must
be stored _______ cooked foods.
5. Cold temperatures slow down the growth of microorganisms.
  TRUE   FALSE
6. Food for storage must be kept covered and/or stored in vermin-proof
containers.   TRUE   FALSE
7. Ice intended for human consumption can be used for storing cans and
bottles.   TRUE   FALSE
8. When foods are stored directly in ice, the water from that ice must be
drained constantly.   TRUE   FALSE
HAZARDS TO OUR HEALTH
fo d P R O T E C T I O N T R A I N I N G M A N U A L
FOOD ALLERGIES
12
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
An allergy is a reaction to a food
or ingredient that the body
mistakenly believes to be harmful.
Millions of Americans suffer from
allergic reactions to food each year.
Most of these food allergies are
mild in nature, but some food
allergies can cause severe reactions,
and may even be life-threatening.
There is no cure for food allergies.
Avoidance of food allergens and
early recognition and management
of allergic food reactions are crucial
to prevent serious health consequences.
Common Symptoms
Following are some of the common
symptoms:
Mild
• Itching
• Wheezing
• Hives
• Swelling of face and eyes
Severe
• Loss of consciousness due to
air way obstruction
• Death
Eight Most Common Allergens
Although an individual could be
allergic to any food product, such as
fruits, vegetables, and meats, however,
the following eight foods account for
90% of all food-allergic reactions:
• Fish
• Peanuts
• Wheat
• Soy
• Tree Nuts
• Eggs
• Milk
• Shell Fish
Here’s an easy way to remember them:
Food Problems Will Send The EMS
These eight foods as well as any
food that contains proteins from
one or more of these foods are
called “major food allergens” by law.
Additives that Trigger Allergies
In addition to the foods listed
above, some common additives of
foods can also trigger an allergic
reaction. Full disclosure of these on
the menu is necessary. Following are
some of the common food additives
used in the food industry:
  Nitrites*—added in meats for
redness.
  Sulfites*—added to dried and
preserved fruits and vegetables for
freshness.
  MSG – added to enhance the
flavor of food.
* The use of Nitrites and Sulfites in the
retail food industry is not permitted.
Hidden Ingredients
Sometimes a dish may contain a
very insignificant amount of common
allergens and only the chef may be
aware of it. Never guess! Always ensure
that a dish is 100% free of allergens.
Review the ingredients list for every
dish requested by the customers and
check labels on packaged and readyto-
eat food products.
Customer Safety
In order to protect the customers,
it is important that there is full disclosure
of the use of these eight
common allergens to the customers.
This can be done in the following
manner:
  By describing details of menu
items.
  When uncertain about any
ingredient, inform the customer
immediately.
  Ensure that food has no contact
with ingredients to which customer
is allergic. Even the smallest
amount of allergen can cause a
serious reaction.
  Wash hands thoroughly and use
new sanitary gloves before
others develop toxins during storage.
Puffer fish may contain
tetrodotoxin and/or saxitoxin which
can cause severe illness and death.
These are central nervous system
toxins and according to FDA, are
1,200 times more deadly than
cyanide.
Certain predatory fish, such as
the barracuda, feed on smaller fish
that had been feeding on algae.
Algae, during certain seasons and in
certain waters may be toxic.
This toxicity accumulates in the
smaller fish and then in the fish
that eat the smaller fish. In this
manner the ciguatoxin, which is
not destroyed by cooking, may
accumulate in fish and this leads
to the illness Ciguatera.
Scombroid poisoning is a food
borne illness caused by the consumption
of marine fish from the
Scombridae family: tuna, mackerel,
and a few non-Scombroidae relatives,
such as bluefish, dolphin and
amberjacks. These fish have high
levels of histidine in their flesh and
during decomposition, the histidine
is converted into histamine which
causes consumers to suffer an allergic-
like reaction. The symptoms of
this illness, among other things,
mimic a heart attack. Histamine is
not destroyed by cooking.
13
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
preparing dishes for guests with
food allergy.
  Clean and sanitize all equipment,
cooking and eating utensils, and
food contact surfaces with hot
soapy water before preparing
allergen-free foods.
  Never use any equipment or
utensils previously used to cook
other foods.
  Never cook with oils that were
used to prepare other foods. Heat
does not destroy allergens.
  Look out for splashes and accidental
spills.
It is important to remember that
removing allergens from a finished
dish, such as nuts, shellfish etc, does
not make the dish safe.
If a guest has an allergic reaction,
call 911 immediately. To prevent
future mistakes, find out what
went wrong.
In order to understand the reasons
behind food sanitation practices, it
is necessary to know a few facts about
the microorganisms which cause food
spoilage and foodborne disease.
Bacteria
Bacteria, commonly called germs,
are extremely small, plant-like
organisms which must be viewed
through a microscope in order to be
seen. If 25,000 bacteria are placed
in a line, that line would only be
one inch long; one million could fit
on the head of a pin. Like any living
thing, bacteria require food, moisture
and the proper temperature for growth.
Most of them need air (these are called
aerobes), but some can survive only in
the absence of air (these are called
anaerobes) and some can grow with or
without air (these are called facultative).
Bacteria are found everywhere on
the earth, in the air and in water.
Soil abounds with bacteria which
grow on dead organic matter.
Shapes of Bacteria
One method of classifying bacteria
is by their shape. All bacteria can be
assigned to one of the following categories:
  Cocci are round or spherical in
shape. While they are able to live
alone, they often exist in groups.
Single chains are called streptococci.
Those which form a grape-like
cluster are called staphylococci
while those that exist in pairs are
called diplococci.
  Bacilli are rod shaped. Some of
these also congregate in the single
chain form and are called streptobacilli.
  Spirilla are spiral or comma
shaped.
Spores
Some bacteria are able to protect
themselves under adverse conditions
by forming a protective shell or wall
around themselves; in this form they
are in the non-vegetative stage and
are called spores. These bacterial
spores can be likened to the seeds of
a plant which are also resistant to
adverse conditions.
During the spore stage bacteria
do not reproduce or multiply. As
soon as these spores find themselves
under proper conditions of warmth,
moisture and air requirement, they
resume their normal vegetative stage
and their growth . Since spores are
designed to withstand rigorous conditions,
they are difficult to destroy
by normal methods. Much higher
killing temperatures and longer time
periods are required. Fortunately,
MICROBIOLOGY OF FOODS — BACTERIA
QUICK REVIEW
The presence of the following in food constitutes a physical hazard:
1. Pieces of glass   TRUE   FALSE
2. Metal shavings   TRUE   FALSE
3. Piece of wood   TRUE   FALSE
4. Pebbles and stones   TRUE   FALSE
5. MSG   TRUE   FALSE
6. Toothpick   TRUE   FALSE
The presence of the following in the food constitutes a chemical hazard:
7. Ciguatoxin   TRUE   FALSE
8. Prescription medicines   TRUE   FALSE
9. Roach spray   TRUE   FALSE
10. Hair   TRUE   FALSE
11. False fingernails   TRUE   FALSE
12. Hair dye   TRUE   FALSE
13. Sulfites can be used in food preparation as long as their use is disclosed
on the menu.   TRUE   FALSE
14. Some wild mushrooms can be very toxic; therefore mushrooms must
always be purchased from a reliable and trustworthy commercial source.
  TRUE   FALSE
15. Use of MSG ( Monsodium Glutamate) in foods is a very dangerous practice
and is not allowed under any circumstances.   TRUE   FALSE
14
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
there are only a relatively few pathogenic
or disease causing bacteria
which are spore-formers. Tetanus,
anthrax and botulism are diseases
caused by spore-formers.
Bacterial Reproduction
Bacteria reproduce by splitting in
two; this is called binary fission. For
this reason, their numbers are always
doubling: one bacterium generates two;
each of these generates another two
resulting in a total of four and the
four become eight and this goes on
and on.
The time it takes for bacteria to
double (generation time) is roughly
twenty to thirty minutes under
favorable conditions.
Types of Bacteria According to
Their Effect on Humans
Types of bacteria classified
according to their effect on us are:
  Harmful or disease-causing bacteria
are known as pathogenic
bacteria or pathogens. They cause
various diseases in humans,
animals and plants.
  Undesirable bacteria which cause
decomposition of food are often
referred to as spoilage bacteria.
  Beneficial bacteria are used in
the production of various foods
including cultured milk, yogurt,
cheese and sauerkraut.
  Benign bacteria, as far as we
know at the present time, are
neither helpful nor harmful to
humans. Of the hundreds of
thousands of strains of bacteria,
most fall in this category.
It must be realized that many
bacteria are essential in the balance
of nature thus the destruction
of all bacteria in the world
would be catastrophic. Our main
objective is public health protection
through the control and
destruction of pathogenic (disease
causing) bacteria and those that
cause food spoilage.
Bacterial Growth
Bacteria require certain conditions
in order to multiply. They need
moisture, warmth, nutrients and time.
It is rapid bacterial multiplication
that often causes problems with
regard to the safety of a food product.
Under ideal conditions rapid
growth can mean that one organism
can become two in as little as
20–30 minutes.
The Bacterial Growth Curve table
assumes that a certain food initially
contains 1,000 organisms. The ideal
rapid growth takes place during the
log phase and all bacteria will reach
this rapid part of their growth if
given the correct conditions.
Bacteria begin their growth cycle by
adjusting to any new environment or
condition by being in a resting or
lag phase. Stationary and death
phases are usually brought about by
the depletion of available
nutrients and the production
of their waste.
Conditions Necessary
for the Growth of
Bacteria (FATTOM)
  Food—Bacteria require
food for growth. The
foods that they like the
most are the same ones we
do. These are generally
high protein foods of animal
origin, such as meat, poultry, fish,
shellfish, eggs, milk and milk
products. They also love plant
products that are heat treated, such
as cooked potato, cooked rice,
tofu, and soy protein foods.
  Acidity—Bacteria generally prefer
neutral foods. They do not fare
well in foods that are too acidic
or too alkaline. This is why vinegar
is used as a preservative.
Acidity is measured in pH. Any
food with a pH value of 4.6 or
less is considered too acidic for
bacteria to grow, therefore, these
foods are relatively safer.
1:00
10
100
1,000
10,000
100,000
1,000,000
1,000,E+07
3:00 5:00 7:00 9:00 11:00 1:00 3:00 5:00
Lag
pm am
Log
Death
Stationary
l
Growth Phases
Bacterial Growth Curve
Growth of Bacteria
Number of
Time Organisms
30 minutes later 2,000
1 hour later 4,000
11/2 hours later 8,000
2 hours later 16,000
21/2 hours later 32,000
3 hours later 64,000
31/2 hours later 128,000
4 hours later 256,000
pH Values of Some Popular Foods
Approximate
Product pH range
Ground beef 5.1 to 6.2
Ham 5.9 to 6.1
Fish (most species) 6.6 to 6.8
Clams 6.5
Oysters 4.8 to 6.3
Crabs 7.0
Butter 6.1 to 6.4
Buttermilk 4.5
Cheese 4.9 to 5.9
Milk 6.3 to 7.0
Yogurt 3.8 to 4.2
Vegetables 3.1 to 6.5
Fruits 1.8 to 6.7
Orange juice 3.6 to 4.3
Melons 6.3 to 6.7
Mayonnaise 3.0 to 4.1
(commercial)
  Temperature—In general, bacteria
prefer warm temperatures. Those
that prefer our food grow
between 41–140°F (Temperature
Danger Zone). This temperature
range includes normal body temperature
and usual room temperature.
However, different types of
bacteria prefer different temperatures.
Mesophilic Bacteria grow best at
temperatures between 50–110°F.
Most bacteria are in this group.
Thermophilic Bacteria prefer
heat and grow best at temperatures
between 110–150°F or more.
Psychrophilic Bacteria prefer cold
and grow at temperatures below 50°F.
One way to control the growth of
bacteria is to ensure that they are
not within the Temperature Danger
Zone (See Page 2).
  Time —Bacteria require time to
grow and multiply. When conditions
are favorable, one bacterium
will split and become two every
twenty to thirty minutes. Thus,
the more time they have, the more
bacteria will be produced. The
simplest way of controlling bacteria
is to minimize the time foods
stay in the temperature danger
zone.
  Oxygen—Some bacteria need
oxygen from the air in order to
grow; these are called aerobes.
Others prefer it when there is no
air or oxygen; these are called
anaerobes. There are yet others
that will thrive whether oxygen is
present or not; these are called
facultative aerobes or facultative
anaerobes.
  Moisture—Bacteria need moisture
or water in order to survive.
Food is absorbed in a liquid form
through the cell wall. If moisture
is not present in sufficient quantity,
bacteria will eventually die.
Bacteria can be controlled by
removing moisture from
foods by the processes of
dehydration, freezing and
preserving in salt or sugar.
The amount of moisture in
a food is measured by Water
Activity value. Any food
with a Water Activity value
of .85 or less does not have
enough moisture to support
the active growth of bacteria.
Locomotion
Bacteria cannot crawl, fly or
move about. A few types do
have thread-like appendages
called flagella with which they
can propel themselves to a very limited
extent. Therefore they must be
carried from place to place by some
vehicle or through some channel.
The modes of transmission
include: air, water, food, hands,
coughing, sneezing, insects, rodents,
dirty equipment, unsafe plumbing
connections and unclean utensils.
Hands are one of the most dangerous
vehicles. There is no doubt that
that if food workers would take better
care of their hands then the incidence
of foodborne disease would
be reduced greatly.
Destruction by Heat
The most reliable and time-tested
method of destroying bacteria is the
use of heat. This method is most
effective when both time and temperature
factors are applied. In
other words, not only do we have to
reach the desired temperature to
destroy bacteria, but we must allow
sufficient time to permit the heat to
kill the more sturdy ones. The lower
the temperature of the heat applied,
the longer the time required to kill
bacteria; conversely, the higher the
temperature, the less time is necessary.
An example of this principle
involves the two accepted methods
for pasteurizing milk. In the “holding”
method, milk is held at a temperature
of 145°F for thirty minutes
while in the “flash” or “high temperature
short time” method, milk
is held at 161°F for fifteen seconds.
Destructon by Chemicals
Bacteria can be destroyed by
chemical agents. Chemicals that kill
bacteria are called germicides or
bactericides. Examples are carbolic
acid, formaldehyde, iodine, chlorine
and quaternary compounds. The
effectiveness of a bactericide
depends on the concentration used.
When used to kill pathogenic (disease-
causing) organisms, it is called
a sanitizer. The most popular sanitizer
used in the food industry is
chlorine.
Other Methods of Destruction
When exposed to air and sunlight,
bacteria are destroyed due to
the combined effects of the lack of
moisture and exposure to the ultraviolet
rays of the sun.
15
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Water Activity of Some Popular Foods
Water
Food Activity
Fresh fruits .97 to 1.0
Pudding .97 to .99
Bread .96 to .97
Cheese .95 to 1.0
Fresh meat .95 to 1.0
Cakes .90 to .94
Cured meat .87 to .95
Jam .75 to .80
Honey .54 to .75
Dried fruit .55 to .80
Chocolate candy .55 to .80
Caramels .60 to .65
Dried milk .20
Dried vegetables .20
Crackers .10
16
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Viruses
Viruses are minute organic forms
which seem to be intermediate
between living cells and organic
compounds. They are smaller than
bacteria, and are sometimes called
filterable viruses because they are so
small that they can pass through the
tiny pores of a porcelain filter which
retain bacteria. They cannot be seen
through a microscope (magnification
of 1500x), but can be seen
through an electron microscope
(magnification of 1,000,000x).
Viruses cause poliomyelitis, smallpox,
measles, mumps, encephalitis,
influenza, and the common cold.
Viruses, like bacteria, are presumed
to exist everywhere.
Unlike bacteria, viruses cannot
reproduce in the food. Food only
serves as a reservoir and a transporting
mechanism until it is
ingested. Once viruses invade our
body, they use our cells to duplicate
themselves. Most often, the presence
of viruses in food supply is an indication
of contamination through
human feces. Food worker’s poor
personal hygiene – for instance, not
washing hands thoroughly after
using the toilet, is a major cause of
these viral infections. The two most
common types of viruses in the
food industry are Hepatitis A, and
Noroviruses (previously known as
Norwalk Virus). Noroviruses have
been recently implicated in various
food borne illness outbreaks involving
cruise ships. Noroviruses are
highly contagious and can spread
very quickly. Hepatitis A virus can
be fatal as it affects the liver.
Parasites
Parasites are organisms that live
in or on other organisms without
benefiting the host organisms. Parasites
are not capable of living independently.
The two most common parasites
that affect the food industry include
trichinella spiralis, which is commonly
associated with pork, and the
round Anisakid worm that is associated
with many species of fish.
With the growing interest in eating
raw marinated fish such as sushi, sashimi,
ceviche etc., there is an increased
risk of illnesses such as Anisakiasis.
Yeasts
Yeasts are one-celled organisms
which are larger than bacteria.
They, too, are found everywhere,
and require food, moisture,
warmth, and air for proper growth.
Unlike some bacteria which live without
air, yeasts must have air in order to
grow. They need sugar, but have the
ability to change starch into sugar.
When yeasts act on sugar, the formation
of alcohol and carbon dioxide
results. In the baking industry,
yeast is used to “raise dough”
through the production of carbon
dioxide. The alcohol is driven off by
the heat of the oven. In wine production,
the carbon dioxide gas bubbles
off, leaving the alcohol. The
amount of alcohol produced by
yeasts is limited to 18%, because
yeasts are killed at this concentration
of alcohol.
Yeasts reproduce by budding, which
is similar to binary fission. Generally,
the methods described for destruction
of bacteria will kill yeasts as well.
Yeasts are not generally considered
to be pathogenic or harmful,
although a few of them do cause
skin infections. Wild yeasts, or those
that get into a food by accident
rather than by design of the food
processor, cause food spoilage and
decomposition of starch and sugar,
and therefore are undesirable.
Molds
Molds are multicellular (manycelled)
microscopic plants which
become visible to the naked eye
when growing in sufficient quantity.
Mold colonies have definite colors
(white, black, green, etc.). They are
larger than bacteria or yeasts. Some
molds are pathogenic, causing such
diseases as athletes’ foot, ringworm,
and other skin diseases. However,
moldy foods usually do not cause
illness. In fact, molds are encouraged
to grow in certain cheeses to produce
a characteristic flavor.
The structure of the mold consists
of a root-like structure called the
mycelium, a stem (aerial filament)
called the hypha, and the spore sac,
called the sporangium. All molds
reproduce by means of spores. Molds
are the lowest form of life that have
these specialized reproductive cells.
Molds require moisture and air
for growth and can grow on almost
any organic matter, which does not
necessarily have to be food. Molds
do not require warmth, and grow
Refrigeration
Refrigeration of foods does not
destroy the bacteria already present.
Cold temperatures from 0°F to
41°F will inhibit or slow the growth
of bacteria. Thus, a food item will
still be safe after several days in a
refrigerator but not indefinitely.
Freezing foods at or below 0°F
will further slow or even stop the
growth of bacteria but will not kill
them.
MICROBIOLOGY OF FOODS — OTHER MICROORGANISMS
17
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
very well in refrigerators. Neither do
molds require much moisture,
although the more moisture present,
the better they multiply.
Methods of destruction for molds
are similar to those required for bacteria.
Heat, chemicals, and ultraviolet
rays destroy mold spores as well
as the molds. Refrigeration does not
necessarily retard their growth.
Certain chemicals act as mold
inhibitors. Calcium propionate
(Mycoban) is one used in making
bread. This chemical when used in
the dough, retards the germination of
mold spores, and bread so treated will
remain mold-free for about five days.
One of the most beneficial molds
is the Penicillium mold from which
penicillin, an antibiotic, is extracted.
The discovery, by Dr. Alexander
Fleming, of the mold’s antibiotic
properties opened up a whole field
of research, and other antibiotic
products from molds have been
discovered.
There are three categories of
foodborne illnesses: infection,
intoxication and toxin mediated
infection.
Foodborne Infection
This is an illness that is caused by
eating a food that has large numbers
of microorganisms on it. These microorganisms
enter the human digestive
tract and disrupt the functions of the
intestines resulting in diarrhea and
other problems. The severity of the
problem depends on the dosage
ingested and the particular bacterium.
The first symptoms of an infection
will occur from as early as six hours to
as long as forty eight hours after the
contaminated food is eaten.
Foodborne Intoxication
This is an illness that is caused by
eating a food that has the toxins that
are generated by certain microorganisms.
The longer a micro-organism
is on a food, the more time it has
to multiply and produce its waste
products. These waste products are
toxins and result in an intoxication
when that food is eaten.
It is important to note that an
intoxication will cause nausea and
vomiting, either immediately after
the food is eaten or within the first
six hours. Also, toxins are not
destroyed by heat so once they are
formed no amount of cooking afterwards
will inactivate them.
Foodborne Toxin Mediated
Infection
This illness occurs when one ingests
a food that has microorganisms on
it. These micro- organisms find
favorable conditions to grow in the
intestines and produce their toxins
which will then cause a foodborne
illness.
FOODBORNE ILLNESSES
QUICK REVIEW
1. Foods that have been contaminated with pathogenic bacteria (  will   will not) change in taste and smell.
2. Under favorable conditions bacteria can double their population every 20 to 30 minutes.   TRUE   FALSE
3. At what temperature is rapid growth of pathogenic bacteria possible?   65°F   140°F
4. What are the six factors that affect the growth of bacteria?_______, _______, _______, ________, ________, ________.
5. Which of the following foods may encourage rapid growth of bacteria?: Cooked rice/Hard boiled air cooled shell egg
6. What type of bacteria grows best at temperatures between 50°-110°F? _______________
7. What is the water activity level at which bacteria have difficulty reproducing? _______________
8. In the life cycle of bacteria, during which phase do bacteria grow most quickly? _______________
9. Most viral food-borne diseases are the result of poor personal hygiene.   TRUE   FALSE
10. The food-borne parasite typically found in under-cooked pork is: _____________
11. A food-borne parasite typically found in marine fish is: _______________
12. The most popular chemical sanitizer is _______________
13. Food held under refrigeration must be at or below: __________°F
14. The reason for refrigerating potentially hazardous foods is to: _______________
18
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
ILLNESS: Salmonellosis
BACTERIA: Salmonella enteritidis
SOURCE: Animals, poultry, eggs, and humans
FOODS INVOLVED: Chicken, other poultry, eggs
ONSET TIME: 6–48 hours
TYPE OF ILLNESS: Infection
SYMPTOMS: Abdominal pain, diarrhea, chills, fever,
nausea, vomiting, and malaise
CONTROL MEASURES
• Cook chicken, poultry and stuffing to 165°F for at
least 15 seconds.
• Refrigerate raw chicken, poultry, and other meats
to 41°F or lower.
• Pay close attention to eggs: store eggs in a refrigerator
at 45°F or lower. Cook eggs to 145°F or higher, (or
per customer request), break and cook eggs to order,
and use pasteurized eggs instead of raw eggs if a food
is not going to be cooked to at least 145°F.
• Prevent cross contamination.
ILLNESS: Staphylococcal gastroenteritis
BACTERIA: Staphylococcus aureus
SOURCE: Healthy human beings: in nose,
throat, hair, on infected cuts, bruises,
abscesses and acne.
FOODS INVOLVED: Baked goods, custards, pastry, and
cooked foods traditionally left out at
room temperature: ham, sliced meats
and other foods with low water activity
ONSET TIME: 6–48 hours
TYPE OF ILLNESS: Infection
SYMPTOMS: Abdominal pain, diarrhea, chills,
fever, nausea, vomiting, and malaise
CONTROL MEASURES
• Prevent bare hand contact with ready-to-eat foods.
• Practice good personal hygiene.
• Prevent infected food workers from working. Look
out for any worker that has an infected cut or
wound on the hands or skin.
• Keep all foods at 41°F or below; cool foods rapidly.
ILLNESS: Campylobacteriosis
BACTERIA: Campylobacter jejuni
SOURCE: Poultry, pigs, sheep and cattle
FOODS INVOLVED: Chicken, other poultry, beef, liver
and water
ONSET TIME: 2–10 days
TYPE OF ILLNESS: Infection
SYMPTOMS: Diarrhea (often-times bloody), severe
abdominal pain, fever, anorexia,
malaise, headache and vomiting.
CONTROL MEASURES
• Proper sanitization of equipment in order to prevent
cross contamination.
• Thoroughly cook meat, poultry and poultry products.
• Use only pasteurized milk.
• Use potable water.
ILLNESS: Listeriosis
BACTERIA: Listeria monocytogenes
SOURCE: Soil, infected animals or humans,
and water
FOODS INVOLVED: Unpateurized milk, raw vegetables,
poultry, raw meats, cheese
ONSET TIME: 1 day–3 weeks
TYPE OF ILLNESS: Infection
SYMPTOMS: Low grade fever, flu-like symptoms,
stillbirths, meningitis and encephalitis. *
Fatalities may occur
CONTROL MEASURES
• Cook foods thoroughly and to required minimum
temperatures.
• Use only pasteurized milk and dairy products.
• Thoroughly wash raw vegetables before eating.
• Avoid cross contamination.
• Clean and sanitize all surfaces.
ILLNESS: Shigellosis
BACTERIA: Shigella species
SOURCE: Human
FOODS INVOLVED: Raw produce, moist prepared foods–
tuna, macaroni, potato salads, etc.
ONSET TIME: 1–7 days
TYPE OF ILLNESS: Infection
SYMPTOMS: Abdominal pain, diarrhea bloody
stools and fever.
CONTROL MEASURES
• Practice good personal hygiene with special emphasis
on hand washing, especially after using the toilet.
• Avoid bare hands contact with ready-to-eat foods.
• Rapidly cool foods to 41°F or below.
• Avoid cross contamination.
• Eliminate flies from the facility.
• Clean and sanitize all surfaces.
COMMON FOODBORNE ILLNESSES
19
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
ILLNESS: Botulism
BACTERIA: Clostridium botulinum
SOURCE: Soil, water, intestinal tract of animals
and fish
FOODS INVOLVED: Home-canned foods, smoked and
vacuum packaged fish, garlic in oil,
baked potatoes, and thick stews
ONSET TIME: 12–36 hours
TYPE OF ILLNESS: Intoxication
SYMPTOMS: Gastrointestinal symptoms may precede
neurological symptoms: vertigo,
blurred or double vision, dryness of
mouth, difficulty swallowing, speaking
and breathing, muscular weakness
and respiratory paralysis. This
illness may cause fatalities.
CONTROL MEASURES
• Never use home-canned or home-jarred products.
• Store smoked fish at 38°F or below. Store all vacuum
packaged foods according to manufacturer’s recommended
instructions (time and temperatures).
• Keep commercially prepared garlic and other
herbs in oil refrigerated at all times.
• Avoid cross contamination.
ILLNESS: Scombroid poisoning
BACTERIA: Bacteria that help produce histamine
SOURCE:: Tuna, bluefish, mackerel, bonito, and
mahi mahi
FOODS INVOLVED: Cooked or raw tuna, bluefish, mackerel,
bonito, and mahi mahi
ONSET TIME: Minutes–2 hours
TYPE OF ILLNESS: Intoxication
SYMPTOMS: Headache, dizziness, nausea, vomiting,
peppery taste, burning sensation in the
throat, facial swelling and stomach aches.
CONTROL MEASURES
• Use a reputable supplier.
• Refuse fish that have been thawed and re-frozen.
Signs that fish have been re-frozen include dried
or dehydrated appearance; excessive frost or ice
crystals in the package; or white blotches (freezer
burns).
• Check temperatures. Fresh fish must be between
32°F and 41°F.
• Thaw frozen fish at refrigeration temperature of
41°F or below.
ILLNESS: Hemorrhagic colitis
BACTERIA: Shiga toxin producing escherichia coli
such as e.coli 0157:h7
SOURCE: Cattle, humans, unpasteurized milk,
untreated water
FOODS INVOLVED: Raw and undercooked ground meats,
fresh produce, unpasteurized milk
and untreated water
ONSET TIME: 12–72 hours
TYPE OF ILLNESS: Intoxication as well as infection
SYMPTOMS: Diarrhea (often bloody), severe
abdominal pain nausea, vomiting,
chills. In children it may complicate
into hemolytic uremic syndrome
(hus), responsible for kidney failure
and blood poisoning.
CONTROL MEASURES
• Cook ground beef and all ground meats to
158°F or higher.
• Cook all foods to required minimum cooking
temperatures.
• Use pasteurized milk.
• Reheat all foods to 165°F within 2 hours.
• Avoid cross contamination.
• Practice good personal hygiene. Wash hands thoroughly
after touching raw foods or after any
activity that may have contaminated them.
ILLNESS: Clostridium perfringens
enteritis
BACTERIA: Clostridium perfringens
SOURCE: Soil, water, gastrointestinal tract of
healthy humans and animals (cattle,
poultry, pigs, and fish)
FOODS INVOLVED: Meat, stews, chilli, gravies, poultry, beans
ONSET TIME: 8–22 hours
TYPE OF ILLNESS: Intoxication as well as infection
SYMPTOMS: Diarrhea and abdominal pain
CONTROL MEASURES
• Rapidly cool meat dishes. (cooling methods are
discussed in detail on pages 28-29.
• Rapidly reheat foods to 165°F within 2 hours.
• Avoid preparing foods days in advance.
• Do not reheat foods on the steam table or any
other hot holding equipment.
• Hold hot foods at 140°F or above.
20
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
ILLNESS: Bacillus cereus gastroenteritis
BACTERIA: Bacillus cereus
SOURCE: Soil and dust, cereal crops
FOODS INVOLVED: Rice, starchy foods–pasta, potatoes,
dry food products, meats, and milk.
ONSET TIME: 30 minutes–5 hours
TYPE OF ILLNESS: Intoxication as well as infection
SYMPTOMS: Nausea, abdominal pain and watery
diarrhea
CONTROL MEASURES
• Do not keep foods at room temperature.
• Rapidly cool meat dishes.
• Rapidly reheat foods to 165°F within 2 hours.
• Serve cooked foods quickly after preparation.
ILLNESS: Vibrio parahaemolyticus
gastroenteritis
BACTERIA: Vibrio parahaemolyticus
SOURCE: Clams, oysters, scallops, shrimp, crabs
FOODS INVOLVED: Raw or partially cooked shellfish
ONSET TIME: 30 minutes–5 hours
TYPE OF ILLNESS: Intoxication as well as infection
SYMPTOMS: Nausea, abdominal pain and watery
diarrhea
CONTROL MEASURES
• Avoid eating raw or undercooked shellfish.
• Purchase seafood from approved sources.
• Keep all seafood refrigerated at 41°F or lower.
• Avoid cross contamination.
ILLNESS: Hepatitis A
VIRUS: Hepatitis A virus
SOURCE: Human feces, fecal contaminated
waters, fecal contaminated produce
FOODS INVOLVED: Raw or partially cooked shellfish,
fruits and vegetables, salads, cold
cuts, water and ice.
ONSET TIME: 15–50 days
SYMPTOMS: Fever, malaise, lassitude,nausea,
abdominal pain and jaundice
CONTROL MEASURES
• Obtain shellfish from approved sources.
• Ensure that food workers practice good personal
hygiene.
• Avoid cross contamination.
• Clean and sanitize food contact surfaces.
• Use potable water.
ILLNESS: Norovirus gastroentritis
VIRUS: Norovirus (aka norwalk-like virus)
SOURCE:: Human feces, fecal contaminated
waters, fecal contaminated produce
FOODS INVOLVED: Ready-to-eat foods such as salads,
sandwiches, baked products,oysters,
fruits and vegetables.
ONSET TIME: 12–48 hours
SYMPTOMS: Fever, vomiting, watery diarrhea,
abdominal pains
CONTROL MEASURES
• Prevent ill food workers from working until fully
recovered.
• Ensure that food workers practice good personal
hygiene.
• Obtain shellfish from approved sources.
• Avoid cross contamination.
• Clean and sanitize food contact surfaces.
• Use potable water.
ILLNESS: Rotavirus gastroenteritis
VIRUS: Rotavirus
SOURCE: Human feces, fecal contaminated
waters, fecal contaminated food
FOODS INVOLVED: Ready-to-eat foods such as salads,
sandwiches, baked products, contaminated
water
ONSET TIME: 1–3 days
SYMPTOMS: Vomiting,watery diarrhea, abdominal
pains and mild fever
CONTROL MEASURES
• Prevent ill food workers from working until fully
recovered.
• Ensure that food workers practice good personal
hygiene.
• Avoid cross contamination.
• Clean and sanitize food contact surfaces.
• Use potable water.
ILLNESS: Astrovirus gastroenteritis
VIRUS: Astrovirus
SOURCE: Human feces, fecal contaminated
food
FOODS INVOLVED: Ready-to-eat foods such as salads,
sandwiches, baked products, contaminated
water.
ONSET TIME: 10–70 hours
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E 21
SYMPTOMS: Vomiting, watery diarrhea, abdominal
pains and mild fever.
Outbreaks are more likely to occur in
daycare and eldercare facilities.
CONTROL MEASURES
• Prevent ill food workers from working until fully
recovered.
• Ensure that food workers practice good personal
hygiene.
• Avoid cross contamination.
• Clean and sanitize food contact surfaces.
• Use potable water.
ILLNESS: Trichinosis
PARASITE: Trichinella spiralis
SOURCE: Pigs, wild game such as bear and walrus
FOODS INVOLVED: Raw and undercooked pork, pork
products, bear , walrus and any other
food products contaminated with
the former.
ONSET TIME: 4–28 days
SYMPTOMS: Gastroenteritis, fever, facial edema,
muscular pains, prostration, and
labored breathing.
CONTROL MEASURES
• Cook pork and pork products to 150°F or higher
for at least 15 seconds.
• Wash, rinse and sanitize equipment used to process
pork and pork products before use.
• Purchase all pork and pork products from
approved suppliers.
ILLNESS: Anisakiasis
PARASITE: Anisakis simplex
SOURCE: Marine fish (saltwater species)
FOODS INVOLVED: Raw, undercooked, or improperly
frozen fish like pacific salmon, mackerel,
halibut, monkfish, herring, flounder,
fluke, cod, haddock, and other fish
used for sushi, sashimi, and ceviche.
ONSET TIME: Within hours
SYMPTOMS: Mild cases include tingling or tickling
sensation in throat, vomiting, or coughing
up worms. Severe cases include debilitating
stomach pains, vomiting, and diarrhea.
CONTROL MEASURES
• Obtain seafood from approved sources.
• Thoroughly cook all seafood to 140°F or higher.
• Only use sushi-grade fish for sushi and sashimi.
• Any fish to be consumed raw should be frozen at
minus 31°F for 15 hours.
ILLNESS: Cyclosporiasis
PARASITE: Cyclospora cayetanensis
SOURCE: Human feces; fecal contaminated water
FOODS INVOLVED: Raw produce, raw milk, water.
ONSET TIME: About a week
SYMPTOMS: Watery diarrhea, mild fever, nausea,
abdominal pains.
CONTROL MEASURES
• Ensure food workers practice good personal hygiene.
• Wash all produce- fruits and vegetables, especially
berries, thoroughly.
• Use potable water.
QUICK REVIEW
1. Salmonella enteritidis is mainly associated with: ___________
2. Food workers sick with an illness that can be transmitted by contact with food or through food should be: ___________
3. We can control the growth of the microorganism clostridium perfringens by _________ ,_________,___________.
4. Ground meats such as hamburgers must be cooked to a minimum temperature of 158°F to eliminate: ___________
5. Clostridium botulinum causes the disease known as botulism.   TRUE   FALSE
6. The microorganism Clostridium botulinum is mainly associated with the following: Smoked fish/tuna fish
7. The following illness has been associated with under-cooked shell eggs: ___________
8. Staphylococcal food intoxication is a common cause of food-borne illness that can be prevented by cooking foods thoroughly.
  TRUE   FALSE
9. Shigellosis can be eliminated by cooking pork to 150°F for 15 seconds.   TRUE   FALSE
10. Scombroid poisoning occurs when someone eats decomposing: ___________
11. Viral Hepatitis is caused by Bacillus cereus.   TRUE   FALSE
12. Escherichia coli O157:H7 is responsible for causing Hemolytic Uremic Syndrome (HUS) among children.   TRUE   FALSE
13. Escherichia coli O157:H7 is mainly associated with ground poultry.   TRUE   FALSE
14. The illness trichinosis is caused by a parasite known as Trichinella spiralis.   TRUE   FALSE
15. To avoid trichinosis, NYC Health Code requires pork to be cooked to a minimum temperature of: ___________
16. Shellfish tags must be kept with the product until it's used up and then filed away for: ___________
17. Raw, marinated or partially cooked fish is made safe by freezing at ______°F for ______
fo d P R O T E C T I O N T R A I N I N G M A N U A L
22 N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Personal hygiene simply means
keeping yourself, and your clothes
as clean as possible. Proper personal
hygiene is extremely important in
preventing food borne illness since
people are the main source of food
contamination. Food workers should
always practice the highest standards
of personal hygiene to ensure that
food is safe from biological, chemical,
and physical hazards. Personal hygiene
enhances the good public image that is
so essential to a good food business.
Highest standards of personal hygiene
include proper hand washing, short
and clean fingernails, notifying supervisor
when ill, use of proper hair
restraints, proper use of disposable
gloves, refraining from wearing jewelry,
avoid eating, drinking, smoking or
otherwise engaging in any activity that
may contaminate the foods.
Personal hygiene is a combination
of several components described below:
Proper Work Attire
Employees who prepare or serve food
products, or wash and sanitize equipment
and utensils must wear clean outer garments.
It is recommended that aprons,
chef jackets, or smocks are worn over
street clothing. Whenever food workers
leave the food area, they should remove
their apron and store it properly. For
example, when using the bathroom, on
breaks, taking out trash, or delivering
food.
Keep personal clothing and other
personal items away from food handling
and storage areas. Employers
must provide adequate storage areas
for employees’ personal belongings.
Hair Restraints
Food workers are required to wear
hair restraints such as hair nets, caps,
hats, scarves, or other form of hair
restraints that are effective (facial hair
included). This is necessary to prevent
them from touching their hair as
well as to prevent hair from falling into
the food.
Wearing of Jewelry
Wearing jewelry such as necklaces,
bracelets, earrings, and other jewelry
while working poses a physical hazard
and as such should not be worn by
food workers when preparing or
serving food (a wedding band is an
exception to this rule.)
Importance of Clean Hands
Clean hands are extremely important
for the safety of food. Most people do
not realize that as part of the normal
flora, we carry a lot of different disease
causing microorganisms on our hands.
For instance, it is estimated that roughly
50–75 % of all healthy humans carry
the Staphylococcus bacteria (mainly in
the nasal passage which can easily be
transferred to hands by simply touching
or blowing the nose). About 60–70%
of the healthy humans carry Clostridium
perfringens, which can also be easily
transmitted onto foods with hands.
In addition to the normal flora, there
are also transient microorganisms
found on our hands that we pick up
through incidental contact by touching
various objects. For instance, traveling
to work from home, we may end up
touching various contaminated surfaces,
e.g., door handles, turnstiles, etc.
This is the reason why hands must be
washed often and thoroughly.
Hand washing
Washing hands properly is the
most effective way of removing
microorganisms. Proper hand washing
involves the use of both hot and
cold running water, soap, and paper
towels or a hot air dryer.
PERSONAL HYGIENE
Use hot and cold running water
Wet hands and apply soap,
lather generously up to the elbow
Rub hands together for 20 seconds
Rinse hands thoroughly
Use a brush on the nails.
Dry hands on disposable paper
towels or a hot air dryer
The Steps of
Proper Hand Washing
fo d P R O T E C T I O N T R A I N I N G M A N U A L
23
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
Always ensure that hands are
washed and dried thoroughly before
starting work, between tasks, and
before working with food products,
equipment, utensils, and linen.
Correct hand washing includes
cleaning the backs of hands, palms,
forearms, between fingers and under
the fingernails using hot water,
soap, and a fingernail brush.
Hand-washing sinks must be
located within 25 feet of each food
preparation, food service and warewashing
area, and in or adjacent to
employee and patron bathrooms.
Doors, equipment and other material
cannot block hand-washing sinks.
Bare hand contact
The New York City Health Code
prohibits the handling of ready-toeat
foods with bare hands. Although
proper hand washing reduces a significant
number of microorganisms
from hands, but never removes all
of them. In addition to that, many
people can also be carriers of disease
causing microorganisms without
getting sick themselves. These individuals
may not show the symptoms
(asymptomatic) or they may have
recovered from an illness, but they
can easily pass these germs to others
through contact with food or food
areas. This is why it is important to
prevent bare hand contact with
ready-to-eat foods by means of sanitary
gloves or other utensils such as
tongs, spatula, deli paper (tissue), or
other utensils.
Exclusion of sick Employees
Any food employee who is sick
with an illness that is transmissible
through contact with food must be
excluded from working in the food
establishment until fully recovered.
Some of these illnesses include:
• Amebiasis
• Cholera
• Cryptosporidiosis
• Diptheria
• E. Coli 0157:H7
• Giardiasis
• Hepatitis A
• Poliomyelitis
• Salmonellosis
• Shigellosis
• Streptococcal sore throat
(including scarlet fever)
• Superficial staphylococcal
infection
• Tuberculosis
• Typhoid
• Yersiniosis
• Infected cut or boil
• Any other communicable disease
It is the employee’s responsibility
to inform the supervisor in case of
an illness, however; supervisors
should be vigilant and observe any
signs that may indicate that the
employee may be sick. Train
employees properly on the hazards
of working while ill with a disease
transmissible through contact with
or through food.
Cuts, Wounds, and Sores
All cuts and wounds that are not
infected on the hands and arms
must be completely covered by a
waterproof bandage. Wear singleuse
gloves or finger cots over any
bandages on the hands and fingers.
The Don’t Habits
1) Don’t smoke or use tobacco in any form while in the
food preparation area.
2) Don’t work when you have a fever, cough, cold, upset
stomach or diarrhea.
3) Don’t store personal medication among food.
4) Don’t work if you have an infected, pus-filled wound.
5) Don’t use a hand sanitizer as a substitute for hand
washing. A hand sanitizer may be used in addition to
proper hand washing.
6) Don’t spit about while preparing food.
Personal Hygiene Checklist
At the beginning of each work day ask yourself the
following questions:
4 Did I shower or take a bath before coming to work?
4 Am I sick with a fever, cold or diarrhea?
4 Do I have any infected cuts or burns?
4 Are my nails clean, trimmed and free from nail polish?
4 Are my apron and clothing clean?
4 Did I remove my jewelry?
4 Am I wearing my hat, cap or hairnet?
QUICK REVIEW
1. As Per New York City Health Code, hands must be washed thoroughly at least 3 times every day.   TRUE   FALSE
2. Sick food workers who can transmit their illness thorough contact with food should be prevented from working until
they are well.   TRUE   FALSE
3. Hands must be washed thoroughly after: ________, __________, ___________, ___________, __________, __________.
4. The NYC Health Code requires hand wash sinks to be readily accessible at all ____________ and ______________.
5. The hand wash sinks must be provided with: ________ and ________ running water, ________ and ________.
6. The NYC Health Code requires that all food workers wear proper hair restraint.   TRUE   FALSE
7. A food worker with an infected cut on his/her hand: ____________________
8. During hand washing hands must be rubbed together for at least: __________________
9. Clean aprons can be used for wiping hands.   TRUE   FALSE
10. Hand sanitizer can be used in place of hand washing during busy periods.   TRUE   FALSE
24 N E W Y O R K C I T Y D E P A R T M E N T O F H E A L T H & M E N T A L H Y G I E N E
This is another step during
which care is needed to maintain
food safety. Preparation refers
to the actions that are necessary
before a food item can be cooked,
or in the case of a food that is
served raw, actions that are necessary
before it can be served.
Thawing
Thawing is also referred to as
defrosting. The Health Code
requires that whole frozen poultry
must be thawed before being
cooked, however, a single portion
may be cooked from a frozen state.
Other potentially hazardous
products should be treated in the
same way: individual portions may
be cooked from a frozen state, while
all others should be thawed before
cooking. It is important to use
methods that will allow the entire
mass to thaw evenly. Any method
that only allows the outside surface
to thaw while the inner portion
remains frozen is unacceptable,
since the outside surface will be in
the danger zone for a prolonged
period of time.
The New York City Health Code
allows the following acceptable
thawing methods:
1) Frozen foods can be removed
from the freezer and stored in a
refrigerator a day or two before they
are needed. In this way the frozen
item will defrost but will not go
above 41°F.
2) Frozen foods may be submerged
under water with the cold
water faucet open and the water
running continuously so that any
loose particles may float and run off.
3) Frozen foods may be thawed
in a microwave oven but this may
only be done if:
  After thawing, the food item is
removed immediately for cooking
in the regular oven or stove.
  The entire cooking process takes
place without interruption in the
microwave oven.
Cutting, Chopping, Mixing,
Mincing, Breading
Any necessary process that will
place a food item within the temperature
danger zone must be controlled.
Preparing or processing the
item in batches will minimize the
amount of time that item is out of
refrigerated storage and the opportunity
for microorganisms to grow.
After preparation, if the food is
not cooked immediately, it must
again be refrigerated until it is ready
for cooking. Care must be taken to
ensure that potentially hazardous
foods are never left out in the temperature
danger zone except for very
short periods during preparation.
Cross contamination
This is a term typically used for
any situation where harmful
microorganisms transfer from a raw
or contaminated food to a cooked
or ready-to-eat food. All raw products,
particularly meat, fish and eggs, have
harmful microorganisms. Therefore,
it is important to keep them separate
from cooked or ready-to-eat foods.
Cross-contamination can happen in
many ways, the following are but a few:
FOOD PREPARATION
Thawing Methods
Refrigerator
Cold Running Water
Microwave
NEW
DEDICATED FOOD-WASHING SINKS
Cross-contamination happens when bacteria from one food spread to another.
This is a common cause of foodborne illnesses. One way to prevent this is to
keep cooked and ready-to-eat foods away from potentially hazardous raw foods,
such as meat, poultry and fish. To reduce the risk of cross-contamination, the
Health Code now requires washing food in:
1. A single-compartment culinary sink used for this purpose only.
2. A dedicated compartment of a multi compartment sink.
3. A food-grade container or colander (if neither of the above is available).
4. Food-washing sinks must be cleaned and sanitized prior to use and after the
washing of raw meat.
5. A sink in which food is washed may not be used as a slop or utility sink or
for hand-washing.
fo d P R O T E C T I O N T R A I N I N G M A N U A L
25
fo d P R O T E C T I O N T R A I N I N G M A N U A L
N E"""
    return data


def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    # SYSTEM_PROMPT = "{CONTEXT} and you can only answer questions based on the data provided below, if you can't find answer, do not hallucinate, just say you can't find answer."
    # Instantiate a Generative AI model here
    prompt = "User: " + question + "\n\n Answer: "
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
