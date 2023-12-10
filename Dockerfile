# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code/sentiment

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install tensorflow_datasets
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Install NLTK
RUN pip install nltk

# Download the NLTK stopwords resource
RUN python -m nltk.downloader stopwords

RUN chmod +x app.py

CMD ["python", "app.py"]