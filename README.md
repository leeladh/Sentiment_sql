# Sentiment_sql


1. Requirements
Python 3.x
Flask
NumPy
BeautifulSoup
Spacy
NLTK
TensorFlow
Keras
MySQL Connector for Python
Environment Variables: user_name, password, database



2. Installation
Clone the Repository: git clone [repository-url]

Install Dependencies: Run pip install -r requirements.txt to install necessary Python packages.

Database Setup: Ensure MySQL is installed and running. Create a database and update the environment variables accordingly.

NLTK Data: Download necessary NLTK data with nltk.download('stopwords') and nltk.download('punkt')


3. Code Overview
First Script: model.py
Text Processing: Includes functions to preprocess text data, such as HTML tag removal, contraction expansion, punctuation removal, lemmatization/stemming, and stopword removal.
LSTM Model: Defines the architecture of the LSTM model and loads pre-trained weights.
Prediction Function: Processes text and predicts sentiment using the LSTM model.
Second Script: Flask Application
Flask Setup: Initializes the Flask app and configures database connection.
Routes: Handles GET and POST requests, processes user input, and logs predictions in the database.
Database Integration: Inserts prediction logs into the MySQL database.

4. Database Schema
Table: prediction_logs
Columns: request_data (TEXT), result (VARCHAR)
