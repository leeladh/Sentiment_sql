from flask import Flask, request, render_template, jsonify
from model import get_result
import mysql.connector
import os

app = Flask(__name__)

# Database Configuration
db_config = {
    'user': os.environ["user_name"],
    'password': os.environ["password"],
    'host': 'localhost',
    'port': '3306',
    'database': os.environ["database"],
    'raise_on_warnings': True
}

# Connect to the database
db = mysql.connector.connect(**db_config)
cursor = db.cursor()

@app.route('/', methods=["POST", "GET"])
def hello():
    if request.method == "POST":
        sentiment = request.form.get('sentiment')
        model = request.form.get('model')
        res = get_result(model, sentiment)
        
        # Log to database
        cursor.execute(
            "INSERT INTO prediction_logs (request_data, result) VALUES (%s, %s)",
            (sentiment, res)
        )
        db.commit()

        return jsonify({'prediction': res})

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
    