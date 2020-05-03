# Disaster-Response-Pipeline
Data ETL Pipeline + NLP ML Pipeline + Implementation into a web app

### Here's the file structure of the project:
- app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

- data

|- disaster_categories.csv  # data to process 

|- disaster_messages.csv  # data to process

|- process_data.py # Run this program to extract, tramsform and load data

|- InsertDatabaseName.db   # database to save clean data to. It will be generated after you run process_data.py

- models

|- train_classifier.py # Run this program to build the NLP machine learning model

|- classifier.pkl  # saved model. It will be genearated after you run train_classifier.py

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Try your message in the Web app!
For example, if you input this message "hungry hungry hungry hungry hungry hungry hungry hungry hungry hungry hungry hungry hungry",
you will see the app returning you the classifications of "Related", "Request", "Aid Related", "Food" and "Direct Report" being True.
