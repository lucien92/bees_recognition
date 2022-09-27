
# from google.colab import drive
# drive.mount('/content/drive')

## Téléchargement de la base de données
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing, layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
#!git clone https://github.com/fabiopereira59/abeilles-cap500

## Chargement des données

IMG_SIZE = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/projet_abeilles/projet_abeilles/abeilles-cap500/cap500/train_cap_andrena',
    labels='inferred',
    label_mode='categorical',
    shuffle = False,
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))
class_names = train_ds.class_names
print(class_names)
nb_classes = len(class_names)
print(nb_classes)


validation_ds = keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/projet_abeilles/projet_abeilles/abeilles-cap500/cap500/val_cap_andrena/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))
## Augmentation de données : Sequence et Albumentations
# !pip uninstall opencv-python-headless==4.5.5.62
# !pip install opencv-python-headless==4.1.2.30
# !pip install -q -U albumentations
# !echo "$(pip freeze | grep albumentations) is successfully installed"
from albumentations import (Compose, Rotate, HorizontalFlip, VerticalFlip, Affine, RandomBrightnessContrast, ChannelShuffle)
import albumentations as A

AUGMENTATIONS_TRAIN = Compose([
    Rotate(limit=[0,100], p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Affine(shear=[-45, 45], p=0.5),
    RandomBrightnessContrast(p=0.5)
])
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2 as cv

class AbeillesSequence(Sequence):
    # Initialisation de la séquence avec différents paramètres
    def __init__(self, x_train, y_train, batch_size, augmentations):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = class_names
        self.batch_size = batch_size
        self.augment = augmentations
        self.indices1 = np.arange(len(x_train))
        np.random.shuffle(self.indices1) # Les indices permettent d'accéder
        # aux données et sont randomisés à chaque epoch pour varier la composition
        # des batches au cours de l'entraînement

    # Fonction calculant le nombre de pas de descente du gradient par epoch
    def __len__(self):
        return int(np.ceil(x_train.shape[0] / float(self.batch_size)))
    
    # Application de l'augmentation de données à chaque image du batch
    def apply_augmentation(self, bx, by):

        batch_x = np.zeros((bx.shape[0], IMG_SIZE, IMG_SIZE, 3))
        batch_y = by
        
        # Pour chaque image du batch
        for i in range(len(bx)):
            class_labels = []
            class_id = np.argmax(by[i])
            class_labels.append(self.classes[class_id])

            # Application de l'augmentation à l'image
            img = cv.imread(bx[i])
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            transformed = self.augment(image=img)
            batch_x[i] = transformed['image']
      
        return batch_x, batch_y

    # Fonction appelée à chaque nouveau batch : sélection et augmentation des données
    # idx = position du batch (idx = 5 => on prend le 5ème batch)
    def __getitem__(self, idx):
        batch_x = self.x_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = self.y_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
           
        batch_x, batch_y = self.apply_augmentation(batch_x, batch_y)

        # Normalisation des données
        batch_x = tf.keras.applications.resnet.preprocess_input(batch_x)
        
        return batch_x, batch_y

    # Fonction appelée à la fin d'un epoch ; on randomise les indices d'accès aux données
    def on_epoch_end(self):
        np.random.shuffle(self.indices1)
# Les images sont stockées avec les chemins d'accès
import numpy as np
print(nb_classes)

x_train = np.array(train_ds.file_paths)
y_train = np.zeros((20067, nb_classes))#rentrer la taille du train

ind_data = 0
for bx, by in train_ds.as_numpy_iterator():
  y_train[ind_data:ind_data+bx.shape[0]] = by
  ind_data += bx.shape[0]
# Instanciation de la Sequence
train_ds_aug = AbeillesSequence(x_train, y_train, batch_size=16, augmentations=AUGMENTATIONS_TRAIN)

# Normalisation des données de validation
import numpy as np
import tensorflow as tf

x_val = np.zeros((2362, IMG_SIZE, IMG_SIZE, 3))#rentrer la taille du valval
y_val = np.zeros((2362, nb_classes))#rentrer la taille du val

ind_data = 0
for bx, by in validation_ds.as_numpy_iterator():
  x_val[ind_data:ind_data+bx.shape[0]] = bx
  y_val[ind_data:ind_data+bx.shape[0]] = by
  ind_data += bx.shape[0]

x_val = tf.keras.applications.resnet.preprocess_input(x_val)
## Création du modèle
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf

### Poids d'imagenet
conv_base = keras.applications.resnet.ResNet101(
    include_top=False,
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=nb_classes,
)

model = keras.Sequential(
    [
        conv_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')
    ]
)
model.summary()
### Poids INat2021
# from tensorflow import keras
# conv_base = keras.models.load_model('drive/MyDrive/Stage2A/INat2021/2623871_resnet50_simclr_v1_inat20_no_top.h5')

# model = keras.Sequential(
#     [
#         conv_base,
#         layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')
#     ]
# )
# model.summary()
## Hierarchical loss
import pandas as pd
import numpy as np

hierarchie = pd.read_csv("/home/lucien/Documents/projet_abeilles/projet_abeilles/abeilles-cap500/csv/hierarchie_especes_andrena.csv")

species = hierarchie["species"].unique()
nb_species = len(species)

genus = list(hierarchie["genus"].unique())
nb_genus = len(genus)

family = list(hierarchie["family"].unique())
nb_family = len(family)

subfamily = list(hierarchie["subfamily"].unique())
nb_subfamily = len(subfamily)

#hierarchie.set_index("species", inplace=True)
data = pd.read_csv("/home/lucien/Documents/projet_abeilles/projet_abeilles/abeilles-cap500/csv/liste_classes_71.csv")
#data.set_index("species", inplace=True)

species_to_genus = np.zeros((nb_genus, nb_species))
genus_to_subfamily = np.zeros((nb_subfamily, nb_genus))
subfamily_to_family = np.zeros((nb_family, nb_subfamily))
for i in range(nb_species):
  nb_images = data.at[i, "0"]
  # species -> genus
  genus_species = hierarchie.at[i, "genus"]
  ind_genus = genus.index(genus_species)
  species_to_genus[ind_genus, i] = 1

  # genus -> subfamily
  subfamily_species = hierarchie.at[i, "subfamily"]
  ind_subfamily = subfamily.index(subfamily_species)
  genus_to_subfamily[ind_subfamily, ind_genus] = 1

  # subfamily -> family
  family_species = hierarchie.at[i, "family"]
  ind_family = family.index(family_species)
  subfamily_to_family[ind_family, ind_subfamily] = 1
from numpy.ma.core import transpose
from keras import backend as K
import math
import tensorflow as tf

# Définition de la fonction de perte
def Hierarchicaloss(species_to_genus, genus_to_subfamily, subfamily_to_family, batch_size, alpha=0.1):

    def weight(height=1):
      return math.exp(-alpha * height)
    
    def species_loss(y_true, y_pred):
      height = 0
      return weight(height) * K.categorical_crossentropy(y_true, y_pred)
  
    def species_to_genus_loss(y_true, y_pred):
      height = 1
      y_true_genus = K.transpose(tf.raw_ops.MatMul(a=species_to_genus, b=tf.cast(y_true, tf.float64), transpose_b=True))
      y_pred_genus = K.transpose(tf.raw_ops.MatMul(a=species_to_genus, b=tf.cast(y_pred, tf.float64), transpose_b=True))
      return weight(height) * K.categorical_crossentropy(y_true_genus, y_pred_genus), y_true_genus, y_pred_genus
    
    def genus_to_subfamily_loss(y_true, y_pred):
      height = 2
      y_true_subfamily = K.transpose(tf.raw_ops.MatMul(a=genus_to_subfamily, b=y_true, transpose_b=True))
      y_pred_subfamily = K.transpose(tf.raw_ops.MatMul(a=genus_to_subfamily, b=y_pred, transpose_b=True))
      return weight(height) * K.categorical_crossentropy(y_true_subfamily, y_pred_subfamily), y_true_subfamily, y_pred_subfamily
    
    def subfamily_to_family_loss(y_true, y_pred):
      height = 3
      y_true_family = K.transpose(tf.raw_ops.MatMul(a=subfamily_to_family, b=y_true, transpose_b=True))
      y_pred_family = K.transpose(tf.raw_ops.MatMul(a=subfamily_to_family, b=y_pred, transpose_b=True))
      return weight(height) * K.categorical_crossentropy(y_true_family, y_pred_family)

    def HIERARCHICAL_loss(y_true, y_pred):
      loss_species = tf.cast(species_loss(y_true, y_pred), tf.float64)
      loss_genus, y_true_genus, y_pred_genus = species_to_genus_loss(y_true, y_pred)
      loss_subfamily, y_true_subfamily, y_pred_subfamily = genus_to_subfamily_loss(y_true_genus, y_pred_genus)
      loss_family = subfamily_to_family_loss(y_true_subfamily, y_pred_subfamily)
      return (loss_species + loss_genus + loss_subfamily + loss_family)/batch_size
   
    # Return a function
    return HIERARCHICAL_loss
loss=[Hierarchicaloss(species_to_genus, genus_to_subfamily, subfamily_to_family, batch_size=16, alpha=0.5)]
## Entraînement du modèle
# Ajout de l'optimiseur, de la fonction coût et des métriques
lr = 1e-3
model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss=loss, metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
# Les callbacks, là où on sauvegarde les poids du réseau

#filepath = path to save the model at the end of each epoch

model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/lucien/Documents/projet_abeilles/model_crop_andrena/model', 
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

#early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#    monitor="val_categorical_accuracy",
#    min_delta=0.01,
#    patience=8,
#    verbose=1,
#    mode="auto")
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1,
                              patience=5, min_lr=0.00001, verbose=1)
history = model.fit(train_ds_aug, epochs=150, validation_data = (x_val, y_val), callbacks=[model_checkpoint_cb, reduce_lr_cb]) #nbre d'epoch de fabio:150