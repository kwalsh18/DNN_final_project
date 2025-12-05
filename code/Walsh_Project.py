# Project Code - Kate Walsh
# Import packages
from tensorflow.keras.layers import Activation
import numpy as np
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers, preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

from bs4 import BeautifulSoup
from collections import Counter

from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

# Set Random Seed for Reproducability 
random_seed = 123
tf.keras.utils.set_random_seed(random_seed)

# Load Data
## Load individual datasets to be used for train/test
### Training data, by condition by ROI
ani_vis_EVC_train = pd.read_csv('data/earlyVC_animate_vis_RDM_all_participants.csv')
ani_sem_EVC_train = pd.read_csv('data/earlyVC_animate_sem_RDM_all_participants.csv')

ina_vis_EVC_train = pd.read_csv('data/earlyVC_inanimate_vis_RDM_all_participants.csv')
ina_sem_EVC_train = pd.read_csv('data/earlyVC_inanimate_sem_RDM_all_participants.csv')

ani_vis_LVC_train = pd.read_csv('data/lateVC_animate_vis_RDM_all_participants.csv')
ani_sem_LVC_train = pd.read_csv('data/lateVC_animate_sem_RDM_all_participants.csv')

ina_vis_LVC_train = pd.read_csv('data/lateVC_inanimate_vis_RDM_all_participants.csv')
ina_sem_LVC_train = pd.read_csv('data/lateVC_inanimate_sem_RDM_all_participants.csv')

### Test data (behavioral), by condition 
ani_vis_test = pd.read_csv('data/ani_vis.csv')
ani_sem_test = pd.read_csv('data/ani_sem.csv')

ina_vis_test = pd.read_csv('data/ina_vis.csv')
ina_sem_test = pd.read_csv('data/ina_sem.csv')

### Remove and save label; OHE Labels
OHE = OneHotEncoder()

#### Training data
##### EVC
ani_vis_EVC_train_Y = ani_vis_EVC_train[['LABEL']]
ani_vis_EVC_train_X =ani_vis_EVC_train.drop(columns=['Participant', 'LABEL'])


ani_sem_EVC_train_Y = ani_sem_EVC_train[['LABEL']]
ani_sem_EVC_train_X =ani_sem_EVC_train.drop(columns=['Participant', 'LABEL'])


ina_vis_EVC_train_Y = ina_vis_EVC_train[['LABEL']]
ina_vis_EVC_train_X =ina_vis_EVC_train.drop(columns=['Participant', 'LABEL'])


ina_sem_EVC_train_Y = ina_sem_EVC_train[['LABEL']]
ina_sem_EVC_train_X =ina_sem_EVC_train.drop(columns=['Participant', 'LABEL'])


##### LVC
ani_vis_LVC_train_Y = ani_vis_LVC_train[['LABEL']]
ani_vis_LVC_train_X =ani_vis_LVC_train.drop(columns=['Participant', 'LABEL'])


ani_sem_LVC_train_Y = ani_sem_LVC_train[['LABEL']]
ani_sem_LVC_train_X =ani_sem_LVC_train.drop(columns=['Participant', 'LABEL'])


ina_vis_LVC_train_Y = ina_vis_LVC_train[['LABEL']]
ina_vis_LVC_train_X =ina_vis_LVC_train.drop(columns=['Participant', 'LABEL'])


ina_sem_LVC_train_Y = ina_sem_LVC_train[['LABEL']]
ina_sem_LVC_train_X =ina_sem_LVC_train.drop(columns=['Participant', 'LABEL'])


#### Test data
ani_vis_test_Y = ani_vis_test[['LABEL']]
ani_vis_test_X =ani_vis_test.drop(columns=['LABEL'])
ani_vis_test_Y = (OHE.fit_transform(ani_vis_test_Y)).toarray()

ani_sem_test_Y = ani_sem_test[['LABEL']]
ani_sem_test_X =ani_sem_test.drop(columns=['LABEL'])
ani_sem_test_Y = (OHE.fit_transform(ani_sem_test_Y)).toarray()

ina_vis_test_Y = ina_vis_test[['LABEL']]
ina_vis_test_X =ina_vis_test.drop(columns=['LABEL'])
ina_vis_test_Y = (OHE.fit_transform(ina_vis_test_Y)).toarray()

ina_sem_test_Y = ina_sem_test[['LABEL']]
ina_sem_test_X =ina_sem_test.drop(columns=['LABEL'])
ina_sem_test_Y = (OHE.fit_transform(ina_sem_test_Y)).toarray()

#### And just get labels for plotting
ani_vis_test_label = np.argmax(ani_vis_test_Y, axis=1)
ani_sem_test_label = np.argmax(ani_sem_test_Y, axis=1)
ina_vis_test_label = np.argmax(ina_vis_test_Y, axis=1)
ina_sem_test_label = np.argmax(ina_sem_test_Y, axis=1)


#### Also save the validation data OHE (same for all ani or ina, 1 per label)
ani_val_Y = ani_vis_test_Y
ina_val_Y = ina_sem_test_Y

##### Standardize test data (Divide by 5, values between 0 - 1)
ani_vis_test_X = ani_vis_test_X / 5 

ani_sem_test_X = ani_sem_test_X / 5

ina_vis_test_X = ina_vis_test_X / 5

ina_sem_test_X = ina_sem_test_X / 5

### Now, I want to concatenate some of the training datasets above to make more training data available for (some of) my research questions
"""
Research Questions:
1. When trained on the neural dissimilarity values, can model predict specific images based on their behavioral dissimilarity (collapsed across condition & ROI)? 
2. Are models trained on visual-neural data better at predicting visual-behavioral image labels  compared to semantic-behavior (collapsed across ROI)? 
3. Can a model trained on early visual cortex predict images in late visual cortex (collapsed across condition)? 
"""
#### Question 1 Data (collapse all)
q1X_ani = pd.concat([
ani_vis_EVC_train_X, ani_sem_EVC_train_X,
ani_vis_LVC_train_X, ani_sem_LVC_train_X
], axis=0, ignore_index=True)

q1X_ina = pd.concat([
ina_vis_EVC_train_X, ina_sem_EVC_train_X,
ina_vis_LVC_train_X, ina_sem_LVC_train_X
], axis=0, ignore_index=True)

q1Y_ani = pd.concat([
ani_vis_EVC_train_Y, ani_sem_EVC_train_Y,
ani_vis_LVC_train_Y, ani_sem_LVC_train_Y
], axis=0, ignore_index=True)

q1Y_ani_val = (
    q1Y_ani
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q1X_ani_val = q1X_ani.loc[q1Y_ani_val.index]

print(q1X_ani_val)

q1Y_ani = (OHE.fit_transform(q1Y_ani)).toarray()

q1Y_ina = pd.concat([
ina_vis_EVC_train_Y, ina_sem_EVC_train_Y,
ina_vis_LVC_train_Y, ina_sem_LVC_train_Y
], axis=0, ignore_index=True)

q1Y_ina_val = (
    q1Y_ina
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q1X_ina_val = q1X_ina.loc[q1Y_ina_val.index]

q1Y_ina = (OHE.fit_transform(q1Y_ina)).toarray()

#### Question 2 Data (collapse ROI)
q2X_ani_sem = pd.concat([
ani_sem_EVC_train_X,
ani_sem_LVC_train_X
], axis=0, ignore_index=True)

q2X_ani_vis = pd.concat([
ani_vis_EVC_train_X, 
ani_vis_LVC_train_X
], axis=0, ignore_index=True)

q2X_ina_sem = pd.concat([
ina_sem_EVC_train_X,
ina_sem_LVC_train_X
], axis=0, ignore_index=True)

q2X_ina_vis = pd.concat([
ina_vis_EVC_train_X,
ina_vis_LVC_train_X
], axis=0, ignore_index=True)

q2Y_ani_sem = pd.concat([
ani_sem_EVC_train_Y,
ani_sem_LVC_train_Y
], axis=0, ignore_index=True)

q2Y_ani_sem_val = (
    q2Y_ani_sem
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q2X_ani_sem_val = q2X_ani_sem.loc[q2Y_ani_sem_val.index]

q2Y_ani_sem = (OHE.fit_transform(q2Y_ani_sem)).toarray()

q2Y_ani_vis = pd.concat([
ani_vis_EVC_train_Y, 
ani_vis_LVC_train_Y
], axis=0, ignore_index=True)

q2Y_ani_vis_val = (
    q2Y_ani_vis
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q2X_ani_vis_val = q2X_ani_vis.loc[q2Y_ani_vis_val.index]

q2Y_ani_vis = (OHE.fit_transform(q2Y_ani_vis)).toarray()

q2Y_ina_sem = pd.concat([
ina_sem_EVC_train_Y,
ina_sem_LVC_train_Y
], axis=0, ignore_index=True)

q2Y_ina_sem_val = (
    q2Y_ina_sem
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q2X_ina_sem_val = q2X_ina_sem.loc[q2Y_ina_sem_val.index]

q2Y_ina_sem = (OHE.fit_transform(q2Y_ina_sem)).toarray()

q2Y_ina_vis = pd.concat([
ina_vis_EVC_train_Y,
ina_vis_LVC_train_Y
], axis=0, ignore_index=True)

q2Y_ina_vis_val = (
    q2Y_ina_vis
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q2X_ina_vis_val = q2X_ina_vis.loc[q2Y_ina_vis_val.index]

q2Y_ina_vis = (OHE.fit_transform(q2Y_ina_vis)).toarray()

#### Question 3 Data (collapse condition)
q3X_ani_EVC = pd.concat([
ani_vis_EVC_train_X, ani_sem_EVC_train_X
], axis=0, ignore_index=True)

q3X_ani_LVC = pd.concat([
ani_vis_LVC_train_X, ani_sem_LVC_train_X
], axis=0, ignore_index=True)

q3X_ina_EVC = pd.concat([
ina_vis_EVC_train_X, ina_sem_EVC_train_X
], axis=0, ignore_index=True)

q3X_ina_LVC = pd.concat([
ina_vis_LVC_train_X, ina_sem_LVC_train_X
], axis=0, ignore_index=True)

q3Y_ani_EVC = pd.concat([
ani_vis_EVC_train_Y, ani_sem_EVC_train_Y
], axis=0, ignore_index=True)

q3Y_ani_EVC_val = (
    q3Y_ani_EVC
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q3X_ani_EVC_val = q3X_ani_EVC.loc[q3Y_ani_EVC_val.index]

q3Y_ani_EVC = (OHE.fit_transform(q3Y_ani_EVC)).toarray()

q3Y_ani_LVC = pd.concat([
ani_vis_LVC_train_Y, ani_sem_LVC_train_Y
], axis=0, ignore_index=True)

q3Y_ani_LVC_val = (
    q3Y_ani_LVC
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q3X_ani_LVC_val = q3X_ani_LVC.loc[q3Y_ani_LVC_val.index]

q3Y_ani_LVC = (OHE.fit_transform(q3Y_ani_LVC)).toarray()

q3Y_ina_EVC = pd.concat([
ina_vis_EVC_train_Y, ina_sem_EVC_train_Y
], axis=0, ignore_index=True)

q3Y_ina_EVC_val = (
    q3Y_ina_EVC
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q3X_ina_EVC_val = q3X_ina_EVC.loc[q3Y_ina_EVC_val.index]

q3Y_ina_EVC = (OHE.fit_transform(q3Y_ina_EVC)).toarray()

q3Y_ina_LVC = pd.concat([
ina_vis_LVC_train_Y, ina_sem_LVC_train_Y
], axis=0, ignore_index=True)

q3Y_ina_LVC_val = (
    q3Y_ina_LVC
    .groupby("LABEL", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=random_seed), include_groups=False)
)

q3X_ina_LVC_val = q3X_ina_LVC.loc[q3Y_ina_LVC.index]

q3Y_ina_LVC = (OHE.fit_transform(q3Y_ina_LVC)).toarray()

## Just need to OHE labels from above
ani_vis_EVC_train_Y = (OHE.fit_transform(ani_vis_EVC_train_Y)).toarray()
ani_sem_EVC_train_Y = (OHE.fit_transform(ani_sem_EVC_train_Y)).toarray()
ina_vis_EVC_train_Y = (OHE.fit_transform(ina_vis_EVC_train_Y)).toarray()
ina_sem_EVC_train_Y = (OHE.fit_transform(ina_sem_EVC_train_Y)).toarray()

ani_vis_LVC_train_Y = (OHE.fit_transform(ani_vis_LVC_train_Y)).toarray()
ani_sem_LVC_train_Y = (OHE.fit_transform(ani_sem_LVC_train_Y)).toarray()
ina_vis_LVC_train_Y = (OHE.fit_transform(ina_vis_LVC_train_Y)).toarray()
ina_sem_LVC_train_Y = (OHE.fit_transform(ina_sem_LVC_train_Y)).toarray()

# Model Definition
## Due to the multitude of questions I want to ask, I will use the same architecture for all questions
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

## Model Compile
model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )
### This architecture is copied and repeated for all questions below ###














# 1. When trained on the neural dissimilarity values, can model predict specific images based on their behavioral dissimilarity (collapsed across condition & ROI)?
## Set # epoch
num_epoch_q1 = 5

## Define Models
Q1_ani_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q1_ani_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Q1_ina_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q1_ina_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

## Fit models
Q1_ani = Q1_ani_model.fit(q1X_ani, q1Y_ani, epochs=num_epoch_q1, validation_data=(q1X_ani_val, ani_val_Y))
Q1_ina = Q1_ina_model.fit(q1X_ina, q1Y_ina, epochs=num_epoch_q1, validation_data=(q1X_ina_val, ina_val_Y))

## Plot fitting
### Animate
Q1_ani_acc_title = "Animate Model Accuracy - Training (Epochs = " + str(num_epoch_q1) + ")"
Q1_ani_loss_title = "Animate Model Loss - Training(Epochs = " + str(num_epoch_q1) + ")"

plt.plot(Q1_ani.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q1_ani.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q1_ani_acc_title)

plt.show()
plt.close()

plt.plot(Q1_ani.history['loss'], label='Training Data Loss')
plt.plot(Q1_ani.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q1_ani_loss_title)

plt.show()
plt.close()

### Inanimate
Q1_ina_acc_title = "Inanimate Model Accuracy - Training (Epochs = " + str(num_epoch_q1) + ")"
Q1_ina_loss_title = "Inanimate Model Loss - Training (Epochs = " + str(num_epoch_q1) + ")"

plt.plot(Q1_ina.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q1_ina.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q1_ani_acc_title)

plt.show()
plt.close()

plt.plot(Q1_ina.history['loss'], label='Training Data Loss')
plt.plot(Q1_ina.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q1_ani_loss_title)

plt.show()
plt.close()

## Evaluate Q1 trained models on behavioral data
### Store model loss/accuracy for each test set
#### Animate Visual - Q1
Ani_Vis_Test_Loss, Ani_Vis_Test_Accuracy = Q1_ani_model.evaluate(ani_vis_test_X, ani_vis_test_Y)
print("Q1 Animate Visual, Loss:", Ani_Vis_Test_Loss)
print("Q1 Animate Visual, Accuracy:", Ani_Vis_Test_Accuracy)

### Store model predictions for test set
ani_vis_pred_q1 = Q1_ani_model.predict(ani_vis_test_X)

### Define predictions, labels, and plotting labels
ani_vis_pred_q1 = np.argmax(ani_vis_pred_q1, axis=1)

### Confusion Matrix 
ani_vis_cm_q1 = confusion_matrix(ani_vis_test_label, ani_vis_pred_q1)
print("Animate Visual Confusion Matrix (Q1):\n", ani_vis_cm_q1, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_vis_cm_q1.shape[0], dtype=bool)
sns.heatmap(
    ani_vis_cm_q1,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_vis_cm_q1,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Animate Visual Confusion Matrix (Q1)')

plt.show()
plt.close()

#### Animate Semantic - Q1
Ani_sem_Test_Loss, Ani_sem_Test_Accuracy = Q1_ani_model.evaluate(ani_sem_test_X, ani_sem_test_Y)
print("Q1 Animate Semantic, Loss:", Ani_sem_Test_Loss)
print("Q1 Animate Semantic, Accuracy:", Ani_sem_Test_Accuracy)

### Store model predictions for test set
ani_sem_pred_q1 = Q1_ani_model.predict(ani_sem_test_X)

### Define predictions, labels, and plotting labels
ani_sem_pred_q1 = np.argmax(ani_sem_pred_q1, axis=1)

### Confusion Matrix 
ani_sem_cm_q1 = confusion_matrix(ani_sem_test_label, ani_sem_pred_q1)
print("Animate Semantic Confusion Matrix (Q1):\n", ani_sem_cm_q1, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_sem_cm_q1.shape[0], dtype=bool)
sns.heatmap(
    ani_sem_cm_q1,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_sem_cm_q1,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Animate Semantic Confusion Matrix (Q1)')

plt.show()
plt.close()

#### Inanimate Visual - Q1
Ani_Vis_Test_Loss, Ani_Vis_Test_Accuracy = Q1_ina_model.evaluate(ina_vis_test_X, ina_vis_test_Y)
print("Q1 Inanimate Visual, Loss:", Ani_Vis_Test_Loss)
print("Q1 Inanimate Visual, Accuracy:", Ani_Vis_Test_Accuracy)

### Store model predictions for test set
ina_vis_pred_q1 = Q1_ina_model.predict(ina_vis_test_X)

### Define predictions, labels, and plotting labels
ina_vis_pred_q1 = np.argmax(ina_vis_pred_q1, axis=1)

### Confusion Matrix 
ina_vis_cm_q1 = confusion_matrix(ina_vis_test_label, ina_vis_pred_q1)
print("Inanimate Visual Confusion Matrix (Q1):\n", ina_vis_cm_q1, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_vis_cm_q1.shape[0], dtype=bool)
sns.heatmap(
    ina_vis_cm_q1,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_vis_cm_q1,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Inanimate Visual Confusion Matrix (Q1)')

plt.show()
plt.close()

#### Inanimate Semantic - Q1
Ani_sem_Test_Loss, Ani_sem_Test_Accuracy = Q1_ina_model.evaluate(ina_sem_test_X, ina_sem_test_Y)
print("Q1 Inanimate Semantic, Loss:", Ani_sem_Test_Loss)
print("Q1 Inanimate Semantic, Accuracy:", Ani_sem_Test_Accuracy)

### Store model predictions for test set
ina_sem_pred_q1 = Q1_ina_model.predict(ina_sem_test_X)

### Define predictions, labels, and plotting labels
ina_sem_pred_q1 = np.argmax(ina_sem_pred_q1, axis=1)

### Confusion Matrix 
ina_sem_cm_q1 = confusion_matrix(ina_sem_test_label, ina_sem_pred_q1)
print("Inanimate Semantic Confusion Matrix (Q1):\n", ina_sem_cm_q1, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_sem_cm_q1.shape[0], dtype=bool)
sns.heatmap(
    ina_sem_cm_q1,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_sem_cm_q1,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Inanimate Semantic Confusion Matrix (Q1)')

plt.tight_layout()
plt.show()
plt.close()















# 2. Are models trained on visual-neural data better at predicting visual-behavioral image labels  compared to semantic-behavior (collapsed across ROI)?
## Set # epoch
num_epoch_q2 = 5

## Define Models
Q2_ani_sem_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q2_ani_sem_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Q2_ani_vis_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q2_ani_vis_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Q2_ina_sem_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q2_ina_sem_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Q2_ina_vis_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q2_ina_vis_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

## Fit models
Q2_ani_sem = Q2_ani_sem_model.fit(q2X_ani_sem, q2Y_ani_sem, epochs=num_epoch_q2, validation_data=(q2X_ani_sem_val, ani_val_Y))
Q2_ani_vis = Q2_ani_vis_model.fit(q2X_ani_vis, q2Y_ani_vis, epochs=num_epoch_q2, validation_data=(q2X_ani_vis_val, ani_val_Y))

Q2_ina_sem = Q2_ina_sem_model.fit(q2X_ina_sem, q2Y_ina_sem, epochs=num_epoch_q2, validation_data=(q2X_ina_sem_val, ina_val_Y))
Q2_ina_vis = Q2_ina_vis_model.fit(q2X_ina_vis, q2Y_ina_vis, epochs=num_epoch_q2, validation_data=(q2X_ina_vis_val, ina_val_Y))

## Plot fitting
### Animate Sem
Q2_ani_sem_acc_title = "Animate Semantic Model Accuracy - Training (Epochs = " + str(num_epoch_q2) + ")"
Q2_ani_sem_loss_title = "Animate Semantic Model Loss - Training(Epochs = " + str(num_epoch_q2) + ")"

plt.plot(Q2_ani_sem.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q2_ani_sem.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q2_ani_sem_acc_title)

plt.show()
plt.close()

plt.plot(Q2_ani_sem.history['loss'], label='Training Data Loss')
plt.plot(Q2_ani_sem.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q2_ani_sem_loss_title)

plt.show()
plt.close()

### Animate Vis
Q2_ani_vis_acc_title = "Animate Visual Model Accuracy - Training (Epochs = " + str(num_epoch_q2) + ")"
Q2_ani_vis_loss_title = "Animate Visual Model Loss - Training(Epochs = " + str(num_epoch_q2) + ")"

plt.plot(Q2_ani_vis.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q2_ani_vis.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q2_ani_vis_acc_title)

plt.show()
plt.close()

plt.plot(Q2_ani_vis.history['loss'], label='Training Data Loss')
plt.plot(Q2_ani_vis.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q2_ani_vis_loss_title)

plt.show()
plt.close()

### Inanimate Semantic
Q2_ina_sem_acc_title = "Inanimate Semantic Model Accuracy - Training (Epochs = " + str(num_epoch_q2) + ")"
Q2_ina_sem_loss_title = "Inanimate Semantic Model Loss - Training (Epochs = " + str(num_epoch_q2) + ")"

plt.plot(Q2_ina_sem.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q2_ina_sem.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q2_ina_sem_acc_title)

plt.show()
plt.close()

plt.plot(Q2_ina_sem.history['loss'], label='Training Data Loss')
plt.plot(Q2_ina_sem.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q2_ina_sem_loss_title)

plt.show()
plt.close()

### Inanimate Visual
Q2_ina_vis_acc_title = "Inanimate Visual Model Accuracy - Training (Epochs = " + str(num_epoch_q2) + ")"
Q2_ina_vis_loss_title = "Inanimate Visual Model Loss - Training (Epochs = " + str(num_epoch_q2) + ")"

plt.plot(Q2_ina_vis.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q2_ina_vis.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q2_ina_vis_acc_title)

plt.show()
plt.close()

plt.plot(Q2_ina_vis.history['loss'], label='Training Data Loss')
plt.plot(Q2_ina_vis.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q2_ina_vis_loss_title)

plt.show()
plt.close()

## Evaluate Q2 trained models on behavioral data
### Store model loss/accuracy for each test set
### Evaluate Semantic Behavior on Model Trained on Semantic Neural
##### Animate
Ani_Sem_Sem_Test_Loss, Ani_Sem_Sem_Test_Accuracy = Q2_ani_sem_model.evaluate(ani_sem_test_X, ani_sem_test_Y)
print("Q2 Animate Semantic Neural -> Animate Semantic Behavior, Loss:", Ani_Sem_Sem_Test_Loss)
print("Q2 Animate Semantic Neural -> Animate Semantic Behavior, Accuracy:", Ani_Sem_Sem_Test_Accuracy)

### Store model predictions for test set
ani_sem_sem_pred_q2 = Q2_ani_sem_model.predict(ani_sem_test_X)

### Define predictions, labels, and plotting labels
ani_sem_sem_pred_q2 = np.argmax(ani_sem_sem_pred_q2, axis=1)

### Confusion Matrix 
ani_sem_sem_cm_q2 = confusion_matrix(ani_sem_test_label, ani_sem_sem_pred_q2)
print("Q2 Animate Semantic Neural -> Animate Semantic Behavior Confusion Matrix:\n", ani_sem_sem_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_sem_sem_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ani_sem_sem_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_sem_sem_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Animate Semantic Neural -> Animate Semantic Behavior')

plt.show()
plt.close()

##### Inanimate
Ani_Sem_Sem_Test_Loss, Ani_Sem_Sem_Test_Accuracy = Q2_ina_sem_model.evaluate(ina_sem_test_X, ina_sem_test_Y)
print("Q2 Inanimate Semantic Neural -> Inanimate Semantic Behavior, Loss:", Ani_Sem_Sem_Test_Loss)
print("Q2 Inanimate Semantic Neural -> Inanimate Semantic Behavior, Accuracy:", Ani_Sem_Sem_Test_Accuracy)

### Store model predictions for test set
ina_sem_sem_pred_q2 = Q2_ina_sem_model.predict(ina_sem_test_X)

### Define predictions, labels, and plotting labels
ina_sem_sem_pred_q2 = np.argmax(ina_sem_sem_pred_q2, axis=1)

### Confusion Matrix 
ina_sem_sem_cm_q2 = confusion_matrix(ina_sem_test_label, ina_sem_sem_pred_q2)
print("Q2 Inanimate Semantic Neural -> Inanimate Semantic Behavior Confusion Matrix:\n", ina_sem_sem_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_sem_sem_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ina_sem_sem_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_sem_sem_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Inanimate Semantic Neural -> Inanimate Semantic Behavior')

plt.show()
plt.close()

### Evaluate Visual Behavior on Model Trained on Visual Neural
##### Animate
Ani_vis_vis_Test_Loss, Ani_vis_vis_Test_Accuracy = Q2_ani_vis_model.evaluate(ani_vis_test_X, ani_vis_test_Y)
print("Q2 Animate Visual Neural -> Animate Visual Behavior, Loss:", Ani_vis_vis_Test_Loss)
print("Q2 Animate Visual Neural -> Animate Visual Behavior, Accuracy:", Ani_vis_vis_Test_Accuracy)

### Store model predictions for test set
ani_vis_vis_pred_q2 = Q2_ani_vis_model.predict(ani_vis_test_X)

### Define predictions, labels, and plotting labels
ani_vis_vis_pred_q2 = np.argmax(ani_vis_vis_pred_q2, axis=1)

### Confusion Matrix 
ani_vis_vis_cm_q2 = confusion_matrix(ani_vis_test_label, ani_vis_vis_pred_q2)
print("Q2 Animate Visual Neural -> Animate Visual Behavior Confusion Matrix:\n", ani_vis_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_vis_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ani_vis_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_vis_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Animate Visual Neural -> Animate Visual Behavior')

plt.show()
plt.close()

##### Inanimate
Ani_vis_vis_Test_Loss, Ani_vis_vis_Test_Accuracy = Q2_ina_vis_model.evaluate(ina_vis_test_X, ina_vis_test_Y)
print("Q2 Inanimate Visual Neural -> Inanimate Visual Behavior, Loss:", Ani_vis_vis_Test_Loss)
print("Q2 Inanimate Visual Neural -> Inanimate Visual Behavior, Accuracy:", Ani_vis_vis_Test_Accuracy)

### Store model predictions for test set
ina_vis_vis_pred_q2 = Q2_ina_vis_model.predict(ina_vis_test_X)

### Define predictions, labels, and plotting labels
ina_vis_vis_pred_q2 = np.argmax(ina_vis_vis_pred_q2, axis=1)

### Confusion Matrix 
ina_vis_vis_cm_q2 = confusion_matrix(ina_vis_test_label, ina_vis_vis_pred_q2)
print("Q2 Inanimate Visual Neural -> Inanimate Visual Behavior Confusion Matrix:\n", ina_vis_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_vis_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ina_vis_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_vis_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Inanimate Visual Neural -> Inanimate Visual Behavior')

plt.show()
plt.close()


### Evaluate Semantic Behavior on Model Trained on Visual Neural
##### Animate
Ani_sem_vis_Test_Loss, Ani_sem_vis_Test_Accuracy = Q2_ani_vis_model.evaluate(ani_sem_test_X, ani_sem_test_Y)
print("Q2 Animate Visual Neural -> Animate Semantic Behavior, Loss:", Ani_sem_vis_Test_Loss)
print("Q2 Animate Visual Neural -> Animate Semantic Behavior, Accuracy:", Ani_sem_vis_Test_Accuracy)

### Store model predictions for test set
ani_sem_vis_pred_q2 = Q2_ani_vis_model.predict(ani_sem_test_X)

### Define predictions, labels, and plotting labels
ani_sem_vis_pred_q2 = np.argmax(ani_sem_vis_pred_q2, axis=1)

### Confusion Matrix 
ani_sem_vis_cm_q2 = confusion_matrix(ani_sem_test_label, ani_sem_vis_pred_q2)
print("Q2 Animate Visual Neural -> Animate Semantic Behavior Confusion Matrix:\n", ani_sem_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_sem_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ani_sem_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_sem_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Animate Visual Neural -> Animate Semantic Behavior')

plt.show()
plt.close()

##### Inanimate
Ani_sem_vis_Test_Loss, Ani_sem_vis_Test_Accuracy = Q2_ina_vis_model.evaluate(ina_sem_test_X, ina_sem_test_Y)
print("Q2 Inanimate Visual Neural -> Inanimate Semantic Behavior, Loss:", Ani_sem_vis_Test_Loss)
print("Q2 Inanimate Visual Neural -> Inanimate Semantic Behavior, Accuracy:", Ani_sem_vis_Test_Accuracy)

### Store model predictions for test set
ina_sem_vis_pred_q2 = Q2_ina_vis_model.predict(ina_sem_test_X)

### Define predictions, labels, and plotting labels
ina_sem_vis_pred_q2 = np.argmax(ina_sem_vis_pred_q2, axis=1)

### Confusion Matrix 
ina_sem_vis_cm_q2 = confusion_matrix(ina_vis_test_label, ina_sem_vis_pred_q2)
print("Q2 Inanimate Visual Neural -> Inanimate Semantic Behavior Confusion Matrix:\n", ina_vis_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_sem_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ina_sem_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_sem_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Inanimate Visual Neural -> Inanimate Semantic Behavior')

plt.show()
plt.close()


### Evaluate Visual Behavior on Model Trained on Semantic Neural
##### Animate
Ani_sem_vis_Test_Loss, Ani_sem_vis_Test_Accuracy = Q2_ani_sem_model.evaluate(ani_vis_test_X, ani_vis_test_Y)
print("Q2 Animate Semantic Neural -> Animate Visual Behavior, Loss:", Ani_sem_vis_Test_Loss)
print("Q2 Animate Semantic Neural -> Animate Visual Behavior, Accuracy:", Ani_sem_vis_Test_Accuracy)

### Store model predictions for test set
ani_sem_vis_pred_q2 = Q2_ani_sem_model.predict(ani_vis_test_X)

### Define predictions, labels, and plotting labels
ani_sem_vis_pred_q2 = np.argmax(ani_sem_vis_pred_q2, axis=1)

### Confusion Matrix 
ani_sem_vis_cm_q2 = confusion_matrix(ani_vis_test_label, ani_sem_vis_pred_q2)
print("Q2 Animate Semantic Neural -> Animate Visual Behavior Confusion Matrix:\n", ani_sem_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_sem_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ani_sem_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_sem_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Animate Semantic Neural -> Animate Visual Behavior')

plt.show()
plt.close()

##### Inanimate
Ani_sem_vis_Test_Loss, Ani_sem_vis_Test_Accuracy = Q2_ina_sem_model.evaluate(ina_vis_test_X, ina_vis_test_Y)
print("Q2 Inanimate Semantic Neural -> Inanimate Visual Behavior, Loss:", Ani_sem_vis_Test_Loss)
print("Q2 Inanimate Semantic Neural -> Inanimate Visual Behavior, Accuracy:", Ani_sem_vis_Test_Accuracy)

### Store model predictions for test set
ina_sem_vis_pred_q2 = Q2_ina_sem_model.predict(ina_vis_test_X)

### Define predictions, labels, and plotting labels
ina_sem_vis_pred_q2 = np.argmax(ina_sem_vis_pred_q2, axis=1)

### Confusion Matrix 
ina_sem_vis_cm_q2 = confusion_matrix(ina_vis_test_label, ina_sem_vis_pred_q2)
print("Q2 Inanimate Semantic Neural -> Inanimate Visual Behavior Confusion Matrix:\n", ina_vis_vis_cm_q2, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_sem_vis_cm_q2.shape[0], dtype=bool)
sns.heatmap(
    ina_sem_vis_cm_q2,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_sem_vis_cm_q2,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Q2 Inanimate Semantic Neural -> Inanimate Visual Behavior')

plt.show()
plt.close()



























# 3. Can a model trained on early visual cortex predict images in late visual cortex (collapsed across condition)? 
## Set # epoch
num_epoch_Q3 = 5

## Define Models
Q3_ani_EVC_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q3_ani_EVC_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Q3_ina_EVC_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(60, activation='softmax')  
])

Q3_ina_EVC_model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

## Fit models
Q3_ani_EVC = Q3_ani_EVC_model.fit(q3X_ani_EVC, q3Y_ani_EVC, epochs=num_epoch_Q3, validation_data=(q3X_ani_LVC, q3Y_ani_LVC))
Q3_ina_EVC = Q3_ina_EVC_model.fit(q3X_ina_EVC, q3Y_ina_EVC, epochs=num_epoch_Q3, validation_data=(q3X_ina_LVC, q3Y_ina_LVC))

## Plot fitting
### Animate
Q3_ani_acc_title = "Animate EVC Model Accuracy - Training (Epochs = " + str(num_epoch_Q3) + ")"
Q3_ani_loss_title = "Animate EVC Model Loss - Training(Epochs = " + str(num_epoch_Q3) + ")"

plt.plot(Q3_ani_EVC.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q3_ani_EVC.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q3_ani_acc_title)

plt.show()
plt.close()

plt.plot(Q3_ani_EVC.history['loss'], label='Training Data Loss')
plt.plot(Q3_ani_EVC.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q3_ani_loss_title)

plt.show()
plt.close()

### Inanimate
Q3_ina_acc_title = "Inanimate EVC Model Accuracy - Training (Epochs = " + str(num_epoch_Q3) + ")"
Q3_ina_loss_title = "Inanimate EVC Model Loss - Training (Epochs = " + str(num_epoch_Q3) + ")"

plt.plot(Q3_ina_EVC.history['accuracy'], label='Training Data Accuracy')
plt.plot(Q3_ina_EVC.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title(Q3_ani_acc_title)

plt.show()
plt.close()

plt.plot(Q3_ina_EVC.history['loss'], label='Training Data Loss')
plt.plot(Q3_ina_EVC.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CCE)')
plt.legend(loc='upper right')
plt.title(Q3_ani_loss_title)

plt.show()
plt.close()

## Evaluate Q3 trained models on behavioral data
### Store model loss/accuracy for each test set
#### Inanimate - Q3
ina_Vis_Test_Loss, ina_Vis_Test_Accuracy = Q3_ina_EVC_model.evaluate(q3X_ina_LVC, q3Y_ina_LVC)
print("Q3 Inanimate EVC -> LVC, Loss:", ina_Vis_Test_Loss)
print("Q3 Inanimate EVC -> LVC, Accuracy:", ina_Vis_Test_Accuracy)

### Store model predictions for test set
ina_vis_pred_Q3 = Q3_ina_EVC_model.predict(q3X_ina_LVC)

### Define predictions, labels, and plotting labels
ina_vis_pred_Q3 = np.argmax(ina_vis_pred_Q3, axis=1)
q3Y_ina_LVC_pred = np.argmax(q3Y_ina_LVC, axis=1)

### Confusion Matrix 
ina_vis_cm_Q3 = confusion_matrix(q3Y_ina_LVC_pred, ina_vis_pred_Q3)
print("Inanimate EVC -> LVC Confusion Matrix (Q3):\n", ina_vis_cm_Q3, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ina_vis_cm_Q3.shape[0], dtype=bool)
sns.heatmap(
    ina_vis_cm_Q3,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ina_vis_cm_Q3,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Inanimate EVC (train) -> LVC (test) Confusion Matrix (Q3)')

plt.show()
plt.close()

#### Animate - Q3
ani_Vis_Test_Loss, ani_Vis_Test_Accuracy = Q3_ani_EVC_model.evaluate(q3X_ani_LVC, q3Y_ani_LVC)
print("Q3 Animate EVC -> LVC, Loss:", ani_Vis_Test_Loss)
print("Q3 Animate EVC -> LVC, Accuracy:", ani_Vis_Test_Accuracy)

### Store model predictions for test set
ani_vis_pred_Q3 = Q3_ani_EVC_model.predict(q3X_ani_LVC)

### Define predictions, labels, and plotting labels
ani_vis_pred_Q3 = np.argmax(ani_vis_pred_Q3, axis=1)
q3Y_ani_LVC_pred = np.argmax(q3Y_ani_LVC, axis=1)

### Confusion Matrix 
ani_vis_cm_Q3 = confusion_matrix(q3Y_ani_LVC_pred, ani_vis_pred_Q3)
print("Animate EVC -> LVC Confusion Matrix (Q3):\n", ani_vis_cm_Q3, "\n")

ax = plt.subplot() 
mask_offdiag = ~np.eye(ani_vis_cm_Q3.shape[0], dtype=bool)
sns.heatmap(
    ani_vis_cm_Q3,
    mask=mask_offdiag,              
    annot=False,
    cmap="Greens",
    linewidths=0.1
)
sns.heatmap(
    ani_vis_cm_Q3,
    mask=~mask_offdiag,             
    cmap="Reds",
    alpha=0.25,
    cbar=False,
    annot=False,
    linewidths=0
)
ax.set_xlabel('True Image Label') 
ax.set_ylabel('Predicted Image Label')

ax.set_title('Animate EVC (train) -> LVC (test) Confusion Matrix (Q3)')

plt.show()
plt.close()

