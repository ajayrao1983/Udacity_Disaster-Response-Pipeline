# Udacity_Disaster-Response-Pipeline
Github link - https://github.com/ajayrao1983/Udacity_Disaster-Response-Pipeline

##Libraries Used:

### Data Reading and Saving in SQLlite database
1) sys
2) pandas
3) create_engine from sqlalchemy

### Reading the data from SQLlite database and training a model
1) sys
2) nltk
3) re
4) numpy
5) pandas
6) sklearn
7) create_engine from sqlalchemy

### Flask Web App
1) json
2) plotly
3) pandas
4) nltk
5) flask
6) create_engine from sqlalchemy

## Folder Structure and Files in Repository
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app. This app reads the data from the DisasterResponse.db in the 'data' folder and classifier.pkl model in the 'models' folder.
data
|- disaster_categories.csv # Has the categories for each message in the 'disaster_messages.csv' file. Categories are what we are trying to predict.
|- disaster_messages.csv # Has the messages to be categorized.
|- process_data.py # This file reads the messages and categories, merges the two dataframes and cleans the data so it is ready to train our model
|- DisasterResponse.db # This is the database to store the cleaned dataframe
models
|- train_classifier.py # Reads the data stored in DisasterResponse.db, tokenizes the messages, trains a multioutput classifier pipeline, prints results and stores the final model
|- classifier.pkl # This is final model that is created once the train_classifier.py code is run
README.md


##Project Description
Following a disaster typcially, disaster response organizations will get millions of communication when they have the least resources to filter and respond to these messages.
Most of the times, it is one in every thousand messages that might be relevant to the disaster response professionals. Different organizations take care of different parts of the problem so it is important to identify the right messages so the right organization can address the issue.
Supervised machine learning can be more accurate than keyword search as keyword water might very few times actually relate to someone looking for fresh drinking water and will also miss people who say they're thirsty but don't use the word 'Water'. However, this is a big gap right now in disaster response contexts.
Figure eight has taken messages from different disasters and relabeled them so there are consistent labels across different disasters. This greatly helps in identifying different trends and to build supervised machine learning models which can help disaster response organizations respond to future disasters.

This web app uses the prelabeled messages to train a multioutput classifier, which is then used to categorize a message to one of 36 categories.

##App Funcationality in more details
The 'data' folder reads 'disaster_categories.csv' and 'disaster_messages.csv' files, cleans them, merges them and saves the data as a SQLlite database in the 'data' folder.
The code in 'models' folder reads the data saved in the previous step, and trains a classifier. The trained model is saved as a pickle file in the 'models' folder.
The app finally reads in the data saved in SQLlite database in the 'data' folder, and the saved pickel model file in the 'models' folder and displays the results.

## Instructions to run the app
The main folder containing the 'data', 'models' and 'app' folders is referred to as 'root' folder.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/
