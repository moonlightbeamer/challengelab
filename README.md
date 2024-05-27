# Building Generative AI Apps Assessment Starting Point

This Python Flask application is a starting point for the Google Building Generative AI Apps Assessment. To complete the assessment you need to implement the `search_vector_database` and `ask_gemini` functions in the `main.py` file. 

## To Run
1. Clone the repo
2. Change to the application folder (it should be the `gen-ai-assessment` folder) and run the following to create a Virtual Environment:

```
pip install virtualenv
virtualenv ~/genai-asessment
source ~/genai-asessment/bin/activate
```

3. Use Pip to install the prequisites

```
pip install -r requirements.txt
```

4. To test the pogram run:

```
python main.py
```

5. If running in Google Cloud Shell, test your app using __Web Preview__ on port 8080. If running on your own computer browse to localhost:8080. 

## Completing the Code

To complete the code, you will have to complete __Task 1__ of the Assessment where the Vector Search and Firestore databases are created.

Open `main.py` in your code editor (__Google Cloud Shell__ is recommended). The following fucntions are not implemented: 
* `search_vector_database`
* `ask_gemin`

There are comments in the code to give you some direction. 


## You can further customize the program using the config.yaml file.

Here is an example. Changing the context will change the bahavior of the model. You can make the model pretend to be anything you want (a bartender, mechanic, whatever). The example below makes the model emulate a Barista. 

Change the YAML file and restart the program

```
app:
 title: "CoffeeBot"
 subtitle: "Your friendly online BaristAI"

palm:
  botname: "CoffeeBot"
  context: "Your name is CoffeeBot. You are a barista and expert on all things related to coffee and tea."
  temperature: 0.8
  max_output_tokens: 1024
  top_p: 0.8
  top_k: 40
```

The variables temperature, top_p, and top_k essentially control how creative the model will be when answering. Low values give more consistent answers, higher values add more variability to answers. 

temperature is in the range 0.0 to 1.0
top_p is in the range 0.0 to 1.0
top_k is in the range 1 to 40

max_output_tokens does just what the name implies. It is in the range 1 to 1024.