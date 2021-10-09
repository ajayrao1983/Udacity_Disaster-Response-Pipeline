# Udacity_Disaster-Response-Pipeline
Github link - https://github.com/ajayrao1983/Udacity_Disaster-Response-Pipeline

##Libraries Used:

### Data Reading and Saving in SQLlite database
sys
pandas
create_engine from sqlalchemy

### Reading the data from SQLlite database and training a model
sys
nltk
re
numpy
pandas
sklearn
create_engine from sqlalchemy

### Flask Web App
json
plotly
pandas
nltk
flask
create_engine from sqlalchemy

##Project Description
This web app is designed to read disaster responses from "Figure Eight"  - https://www.figure-eight.com/, and classifies disaster messages.

##What does the app do?
The 'data' folder reads 'disaster_categories.csv' and 'disaster_messages.csv' files, cleans them, merges them and saves the data as a SQLlite database in the 'data' folder.
The code in 'models' folder reads the data saved in the previous step, and trains a classifier. The trained model is saved as a pickle file in the 'models' folder.
The app finally reads in the data saved in SQLlite database in the 'data' folder, and the saved pickel model file in the 'models' folder and displays the results.
