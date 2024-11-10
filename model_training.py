import pandas as pd
import numpy as np
from keras import Sequential, Model
from keras.layers import TextVectorization, LSTM, Dense, Input, Embedding, GRU
from keras.optimizers import Adam
import tensorflow as tf

# CSV parsing function
def load_phishing_data(file_path, nrows=15000):
    df = pd.read_csv(file_path, usecols=['sender', 'subject', 'body', 'label'], nrows=nrows)
    # Combine text features
    df['text'] = df['sender'] + ' ' + df['subject'] + ' ' + df['body']
    return df['text'].to_numpy(dtype=np.str_), df['label'].to_numpy(dtype=np.int32)

# Prepare the data
def prepare_data(texts, labels, max_tokens=10000, sequence_length=100):
    # Create and adapt text vectorization layer
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=sequence_length,
        output_mode='int'
    )

    vectorizer.adapt(texts)
    
    # Vectorize the text
    X = vectorizer(texts)
    y = np.array(labels, dtype='float32')

    return X, y, vectorizer

# Build the model
def create_model(vocab_size, sequence_length):
    model = Sequential([
        Input(shape=(sequence_length,)),
        Embedding(vocab_size, 32),
        GRU(64, return_sequences=True, recurrent_dropout=0.1),
        GRU(32),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Main execution
def main():
    # Load and prepare data
    texts, labels = load_phishing_data('CEAS_08.csv', 15000)

    # Prepare the data
    X, y, vectorizer = prepare_data(texts, labels, 10000, 200)

    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=0.2, random_state=42)

    # Create and compile model
    model = create_model(vocab_size=10000, sequence_length=200)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    #exit()
    
    # Save the Keras model
    model.save('phishing_model')
    
    # Save the vectorizer configuration
    import json
    vocab = vectorizer.get_vocabulary()
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    with open('vectorizer_config.json', 'w') as f:
        json.dump({
            'vocab': vocab_dict,
            'max_tokens': 10000,
            'sequence_length': 200
        }, f)

if __name__ == "__main__":
    main()