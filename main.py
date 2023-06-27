#  utile pour développer des applications web en Python de manière simple et efficace
from flask import Flask, request, render_template 
# utilisée pour les calculs mathématiques en Python, spécialement pour les tableaux à plusieurs dimensions.
import numpy as np
#  convertir des données textuelles en séquences numériques qui peuvent être utilisées comme entrées pour les réseaux de neurones.
from keras.preprocessing.text import Tokenizer
# permet d'assurer que toutes les séquences de données ont la même longueur et peuvent donc être utilisées comme entrées pour le modèle,ajoutant des zéros à la fin des séquences plus courtes pour s'assurer que toutes les entrées ont la même longueur 
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sequential est une classe de Keras qui permet de définir et créer des modèles de réseaux de neurones séquentiels en permettant d'ajouter les couches du modèle les unes après les autres de manière simple et conviviale
from keras.models import Sequential
# import des couches Embedding[vecteur de densite latente], GRU et Dense(effectuer les calculs de regression) pour le modèle RNN
from keras.layers import Embedding, GRU, Dense 
# stocker et manipuler des données tabulaires, telles que des tableaux de données enregistrés dans des fichiers CSV
import pandas as pd

import keras
import pickle 

# Chargement des données de sentiments préparées dans un DataFrame de Pandas
data = pd.read_csv('yelp.csv')

# Séparation des données en entrée (textes) et étiquettes (sentiments)
texts = data['sentence'].values # tous les textes sont stockés dans le tableau texts
labels = data['label'].values # toutes les étiquettes sont stockées dans le tableau labels

# Tokenisation des textes
tokenizer = Tokenizer(num_words=10000) # définition du tokenizer avec une limite de 10000 mots
tokenizer.fit_on_texts(texts) # entraînement du tokenizer sur les textes
# tokenizer.fit_on_texts(texts) entraîne le tokenizer sur les textes. Cela signifie que le tokenizer analyse les textes et crée un dictionnaire de mots uniques, avec un identifiant unique pour chaque mot.
sequences = tokenizer.texts_to_sequences(texts) # tokenisation des textes
# effectue la tokenisation des textes. Cela signifie que le tokenizer convertit chaque séquence de textes en une séquence d'identificateurs uniques. Les séquences de textes sont remplacées par des séquences d'entiers, où chaque entier représente un mot unique dans le dictionnaire créé lors de l'entraînement.

# Pad sequences pour avoir la même longueur pour tous les textes
padded_sequences = pad_sequences(sequences, maxlen=100) # padding des séquences de tokenisation pour toutes les avoir de la même longueur

# Définition du modèle RNN
model = Sequential([
    Embedding(10000, 128, input_length=100), # première couche : embedding
       GRU(128), # deuxième couche : GRU
     Dense(1, activation='sigmoid') # troisième couche : dense avec une activation sigmoïde pour avoir un output binaire
   ])

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # choix de la fonction de coût, de l'optimiseur et des métriques à suivre lors de l'entraînement

# Entraînement du modèle
history = model.fit(padded_sequences, labels, epochs=10, validation_split=0.2) # entraînement sur les séquences padded et les étiquettes avec 10 epochs et une validation sur 20% des données

# Création de l'application Flask
app = Flask(__name__)
# La variable name est un paramètre optionnel qui représente le nom du module en cours d'exécution. Si ce paramètre est spécifié, Flask utilisera ce nom pour trouver les ressources liées à l'application, telles que les templates de rendu et les fichiers statiques. Si ce paramètre n'est pas spécifié, Flask utilisera le nom du script principal.

# Route principale
@app.route('/')
def index():
    return render_template('Accueil.html') # renvoie du template HTML index

# Route pour la prédiction
@app.route('/', methods=['POST'])
def predict():

    # Récupération du texte entré par l'utilisateur
    text = request.form['text']
    
    # Tokenisation et pad de la séquence
    sequence = tokenizer.texts_to_sequences([text])
    #puis ajoute des zéros pour faire en sorte que la longueur de la séquence soit de 100.
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Prédiction du sentiment
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "positive \U0001f600" if prediction > 0.5  else "negative \U0001F614"
    return render_template('Accueil.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run()