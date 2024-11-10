import json
import tensorflow as tf
from keras.layers import TextVectorization, Embedding, GRU, Dense, Input
from keras import Sequential
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the saved vectorizer configuration
with open('vectorizer_config.json', 'r') as f:
    vectorizer_config = json.load(f)

# Recreate the TextVectorization layer with the loaded vocabulary
vocab = list(vectorizer_config['vocab'].keys())
vocab_size = len(vocab)
sequence_length = vectorizer_config['sequence_length']

# Build and configure the TextVectorization layer
vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=sequence_length,
    output_mode='int'
)
vectorizer.set_vocabulary(vocab)

# Load the trained model
model = tf.keras.models.load_model('phishing_model')

# Create a new model that includes the vectorizer layer
inputs = Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = model(x)  # Pass the vectorized text into the loaded model
model_with_vectorizer = tf.keras.Model(inputs, x)

# The new model can now take raw text input

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Get text input
    text = [data['text']]
    
    # Make a prediction
    prediction = model_with_vectorizer.predict(text)[0][0]  # Assuming binary classification

    # Return the result
    return jsonify({'prediction': float(prediction)})

# Run the Flask app
if __name__ == '__main__':
    app.run()

# Example usage:
#sample_text = ["paypalofficial@gmail.com URGENT ACTION REQUIRED hello dear user your account has been compromised and you must take urgent action and verify your credit card number. If not your account will be disabled in 24 hours. Have good day.", 
#               "Hey man! Long time no see! Hey how are you doing man? I am very excited for our meeting on sunday. Are you doing alright? How is the wife? Blessings, Jim", 
#               "Replica Watches <jhorton@thebakercompanies.com> We have fake Swiss Men's and Ladie's Replica Watches from Rolex to the Popular Panerai Watch More information here",
#               "qydlqcws-iacfym@issues.apache.org,xrh@spamassassin.apache.org [Bug 5780] URI processing turns uuencoded strings into http URI's which then causes FPs,\"http://issues.apache.org/SpamAssassin/show_bug.cgi?id=5780 wrzzpv@sidney.com changed: What    |Removed                     |Added ---------------------------------------------------------------------------- OtherBugsDependingO|                            |5813 nThis|                            | ------- You are receiving this mail because: ------- You are the assignee for the bug, or are watching the assignee."]
#prediction = model_with_vectorizer.predict(sample_text)
#print("Prediction:", prediction)
