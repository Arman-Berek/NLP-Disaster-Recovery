# NLP Disaster Response Pipeline

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [NLTK](http://www.nltk.org/install.html)
- [Flask](https://pypi.org/project/Flask/)

### Overview
In this project, I created an ETL pipeline as well as a machine learning pipeline to classify messages sent during a disaster so that those messages can then be sent to appropriate relief agencies. The ETL pipeline loads and cleans the data and then stores it in a SQLite database. The machine learning pipeline takes the data from the database and applies text processing to it. The cleaned data then gets trained and tuned using GridSearchCV. The results are outputted to the test set and the final model gets exported as a pickle file. The Flask web app allows a user to input a message and see how it gets categorized.

### Files

The files in this repository include:
- app (files for the flask app)
    - templates (files containing html pages)
        - go.html: webpage that shows classification results.
        - master.html: webpage that is the home page and shows some visualizations.
    - run.py: runs the ML pipeline and the Flask app.
- data (files keeping track of data)
    - DisasterResponse.db: SQLite database containing table for the disaster data.
    - disaster_categories.csv: file containing the categories data.
    - disaster_messages.csv: file containing messages data.
    - process_data.py: script for the ETL pipeline.
- models (files for the model)
    - classifier.pkl: pickle file containing the trained model.
    - train_classifier.py: script for the ML pipeline.
- ETL Pipeline Preparation.ipynb: notebook for ETL pipeline.
- ML Pipeline Perparation.ipynb: notebook for ML pipeline

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
