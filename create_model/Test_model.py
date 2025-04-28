# Importing Libraries
import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.patches as mpatches
import sys
import warnings
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Tránh lỗi OOM
        print("✅ GPU CUDA đang được sử dụng:", gpus)
    except RuntimeError as e:
        print("⚠ Lỗi khi bật GPU:", e)
else:
    print("⚠ Không tìm thấy GPU! Đang sử dụng CPU.")

# Loading the list of Chopin's midi files as stream
filepath = "./Data/beeth/"
# Getting midi files
all_midis = []
for i in os.listdir(filepath):
    if i.endswith(".mid"):
        tr = filepath + i
        midi = converter.parse(tr)
        all_midis.append(midi)


#Helping function
def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes

#Danh sach so nut nhac tac duoc tư cac ban nhac chopin
Corpus= extract_notes(all_midis)
print("Total notes in all the Chopin midis in the dataset:", len(Corpus))

print("First fifty values in the Corpus:", Corpus[:50])

def chords_n_notes(Snippet):
    Melody = []
    offset = 0  # Incremental
    for i in Snippet:
        # If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")  # Seperating the notes in chord
            notes = []
            for j in chord_notes:
                inst_note = int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        # pattern is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)
    return Melody_midi


Melody_Snippet = chords_n_notes(Corpus[:100])

print(Melody_Snippet)

Melody_Snippet.write('midi', 'Beeth_Melody_Snippet_model3.mid')

#So lượng tung loai nut trong nhac cua Chopin
count_num = Counter(Corpus)
print("Tong số loai cac nut nhạc khac nhau: ", len(count_num))
print(count_num)


Notes = list(count_num.keys())
Recurrence = list(count_num.values())

#Average recurrenc for a note in Corpus
def Average(lst):
    return sum(lst) / len(lst)
print("Average recurrenc for a note in Corpus:", Average(Recurrence))
print("Most frequent note in Corpus appeared:", max(Recurrence), "times")
print("Least frequent note in Corpus appeared:", min(Recurrence), "time")

# Plotting the distribution of Notes
plt.figure(figsize=(18,3),facecolor="#97BACB")
bins = np.arange(0,(max(Recurrence)), 50)
plt.hist(Recurrence, bins=bins, color="#97BACB")
plt.axvline(x=25 ,color="#DBACC1")
plt.title("Frequency Distribution Of Notes In The Corpus")
plt.xlabel("Frequency Of Chords in Corpus")
plt.ylabel("Number Of Chords")
plt.show()


# Getting a list of rare chords
rare_note = []
for index, (key, value) in enumerate(count_num.items()):
    if value < 25:
        m = key
        rare_note.append(m)

print("Total number of notes that occur less than 100 times:", len(rare_note))


#Eleminating the rare notes
for element in Corpus:
    if element in rare_note:
        Corpus.remove(element)

print("Length of Corpus after elemination the rare notes:", len(Corpus))



# Storing all the unique characters present in my corpus to bult a mapping dic.
# Storing all the unique characters present in my corpus to build a mapping dictionary.
symb = sorted(list(set(Corpus)))

L_corpus = len(Corpus)  # length of corpus
L_symb = len(symb)  # length of total unique characters

# Building dictionary to access the vocabulary from indices and vice versa
mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

print("Total number of characters:", L_corpus)
print("Number of unique characters:", L_symb)
print("Mapping:", mapping)
print("Reverse Mapping:", reverse_mapping)


# Splitting the Corpus in equal length of strings and output target
length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])

L_datapoints = len(targets)
print("Total number of sequences in the Corpus:", L_datapoints)
print("features:", features[:5])
print("targets:", targets[:5])

print("Length of features:", len(features[:5][0]))

# reshape X and normalize
X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)

# one hot encode the output variable
y = tensorflow.keras.utils.to_categorical(targets)

#Taking out a subset of data to be used as seed
X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model("Beeth_Melody_Generator_model1.h5")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_predictions(model, X_seed, y_seed, reverse_mapping=None):
    y_pred = []

    for i in range(len(X_seed)):
        seed = X_seed[i].reshape(1, X_seed.shape[1], 1)
        prediction = model.predict(seed, verbose=0)
        predicted_index = np.argmax(prediction)
        y_pred.append(predicted_index)

    y_true = np.argmax(y_seed, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision (macro): {precision:.4f}")
    print(f"✅ Recall (macro): {recall:.4f}")
    print(f"✅ F1-score (macro): {f1:.4f}")

    return y_true, y_pred

y_true, y_pred = evaluate_model_predictions(model, X_seed, y_seed, reverse_mapping)
print("True labels:", y_true)
print("Predicted labels:", y_pred)
print("leng_labels_True:", len(y_true))
print("leng_labels_Predicted:", len(y_pred))

def Malody_Generator(Note_Count):
    seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    Music = ""
    Notes_Generated = []
    for i in range(Note_Count):
        seed = seed.reshape(1, length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0  # diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(L_symb)
        Notes_Generated.append(index)
        if index in reverse_mapping:
            Music = [reverse_mapping[char] for char in Notes_Generated]
        else:
            print(f"Warning: Index {index} not found in reverse_mapping")
            continue
        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]
    # Now, we have music in form of a list of chords and notes and we want to be a midi file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)
    return Music, Melody_midi


#getting the Notes and Melody created by the model
Music_notes, Melody = Malody_Generator(100)
#To save the generated melody
Melody.write('midi','Beeth_Melody_Generated_model1_test.mid')

