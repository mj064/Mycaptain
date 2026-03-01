# import dependencies
import numpy as np
import sys
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

# load data
with open('text_generator/next_word_predictor.txt', 'r', encoding='utf-8') as f:
    file = f.read()

# Tokenize and preprocess
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(file.lower())

# Create sequences for next-word prediction
seq_length = 5
sequences = []
for i in range(seq_length, len(tokens)):
    seq = tokens[i-seq_length:i+1]
    sequences.append(seq)

# Prepare data for model
vocab = sorted(set(tokens))
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}

X = []
y = []
for seq in sequences:
    X.append([word_to_int[w] for w in seq[:-1]])
    y.append(word_to_int[seq[-1]])

X = np.array(X)
y = to_categorical(y, num_classes=len(vocab))

batch_size = 256
model = Sequential()
model.add(Input(shape=(seq_length, 1)))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(len(vocab), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], seq_length, 1)).astype('float16')
X = X / float(len(vocab))

# Model checkpoint and early stopping
filepath = 'text_generator_weights.keras'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Always train and fine-tune from previous weights
if os.path.exists(filepath):
    print("Loading previous weights for fine-tuning...")
    model.load_weights(filepath)
print("Starting training...")
model.fit(X, y, epochs=50, batch_size=batch_size, shuffle=True, callbacks=[checkpoint, early_stop], verbose=1)


# Top-k sampling with repetition penalty
def sample_top_k(preds, temperature=1.0, k=10, prev_words=None):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # Get top k indices
    top_k_indices = preds.argsort()[-k:][::-1]
    top_k_probs = preds[top_k_indices]
    # Penalize repeated words
    if prev_words:
        for i, idx in enumerate(top_k_indices):
            if int_to_word[idx] in prev_words:
                top_k_probs[i] *= 0.5  # reduce probability for repeated words
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    chosen = np.random.choice(top_k_indices, p=top_k_probs)
    return chosen


# Text generation function with top-k sampling and repetition penalty
def generate_text(seed_text, next_words=20, temperature=1.0, k=10):
    prev_words = tokenizer.tokenize(seed_text.lower())
    # Pad with a special token or zeros if too short
    pad_token = '<PAD>'
    if pad_token not in word_to_int:
        word_to_int[pad_token] = 0
        int_to_word[0] = pad_token
    for _ in range(next_words):
        token_list = prev_words[-seq_length:]
        if len(token_list) < seq_length:
            token_list = [pad_token] * (seq_length - len(token_list)) + token_list
        x_input = np.array([word_to_int.get(w, 0) for w in token_list]).reshape(1, seq_length, 1)
        x_input = x_input / float(len(vocab))
        preds = model.predict(x_input, verbose=0)[0]
        pred_index = sample_top_k(preds, temperature, k, prev_words)
        pred_word = int_to_word[pred_index]
        prev_words.append(pred_word)
    return ' '.join([w for w in prev_words if w != pad_token])


# Interactive auto-suggest function with top-k sampling
def auto_suggest():
    print("Type your phrase. Press Enter to get a suggestion. Type 'exit' to quit.")
    seed_text = input("Start your phrase: ")
    prev_words = tokenizer.tokenize(seed_text.lower())
    pad_token = '<PAD>'
    if pad_token not in word_to_int:
        word_to_int[pad_token] = 0
        int_to_word[0] = pad_token
    while True:
        token_list = prev_words[-seq_length:]
        if len(token_list) < seq_length:
            token_list = [pad_token] * (seq_length - len(token_list)) + token_list
        x_input = np.array([word_to_int.get(w, 0) for w in token_list]).reshape(1, seq_length, 1)
        x_input = x_input / float(len(vocab))
        preds = model.predict(x_input, verbose=0)[0]
        pred_index = sample_top_k(preds, temperature=0.8, k=10, prev_words=prev_words)
        pred_word = int_to_word[pred_index]
        print(f"Suggested next word: {pred_word}")
        user_choice = input("Add suggested word? (y/n/exit): ").strip().lower()
        if user_choice == 'y':
            prev_words.append(pred_word)
            print(f"Current phrase: {' '.join([w for w in prev_words if w != pad_token])}")
        elif user_choice == 'n':
            new_word = input("Type your own word: ")
            prev_words.append(new_word)
            print(f"Current phrase: {' '.join([w for w in prev_words if w != pad_token])}")
        elif user_choice == 'exit':
            break

# Main menu
print("Choose an option:")
print("1. Auto-suggest (interactive)")
print("2. Predict N words from a phrase")
choice = input("Enter 1 or 2: ").strip()
if choice == '1':
    auto_suggest()
elif choice == '2':
    phrase = input("Enter your starting phrase: ")
    n_words = int(input("How many words to predict? "))
    temp = float(input("Temperature (e.g., 1.0 for normal, 0.5 for conservative, 1.5 for creative): "))
    k = int(input("Top-k sampling (e.g., 10): "))
    print(generate_text(phrase, next_words=n_words, temperature=temp, k=k))
else:
    print("Invalid choice.")
