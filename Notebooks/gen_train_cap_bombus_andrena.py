#importations
import os
import cv2
#path to images

base_path_train = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/train_cap"
base_path_evaluate = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/val_cap"
base_path_test = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/test_cap"

#Pour le train on cherche dans tous les dossiers qui contiennetn le mot andrena

try:
    os.mkdir(base_path_train + "_andrena")
except:
    pass

base_path_train_andrena = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/train_cap_andrena"

for folder in os.listdir(base_path_train):
    print(folder)
    if "Andrena" in folder:
        print("on est dans le bon dossier")
        try:
            os.mkdir(base_path_train_andrena + "/" + folder)
        except:
            continue
        for img in os.listdir(base_path_train + "/" + folder):
            #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
            image = cv2.imread(base_path_train + "/" + folder + "/" + img)
            print(image)
            cv2.imwrite(str(base_path_train_andrena) + "/" + str(folder) + "/" + img, image)

#Même chose pour l'evaluation

try:
    os.mkdir(base_path_evaluate + "_andrena")
except:
    pass

base_path_evaluate_andrena = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/val_cap_andrena"

for folder in os.listdir(base_path_evaluate):
    print(folder)
    if "Andrena" in folder:
        print("on est dans le bon dossier")
        try:
            os.mkdir(base_path_evaluate_andrena + "/" + folder)
        except:
            continue
        for img in os.listdir(base_path_evaluate + "/" + folder):
            #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
            image = cv2.imread(base_path_evaluate + "/" + folder + "/" + img)
            cv2.imwrite(str(base_path_evaluate_andrena) + "/" + str(folder) + "/" + img, image)


#Même chose pour le test

try:
    os.mkdir(base_path_test + "_andrena")
except:
    pass

base_path_test_andrena = "/home/lucien/Documents/projet_abeilles/abeilles-cap500/cap500/test_cap_andrena"

for folder in os.listdir(base_path_test):
    print(folder)
    if "Andrena" in folder:
        print("on est dans le bon dossier")
        try:
            os.mkdir(base_path_test_andrena + "/" + folder)
        except:
            continue
        for img in os.listdir(base_path_test + "/" + folder):
            #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
            image = cv2.imread(base_path_test + "/" + folder + "/" + img)
            print(image)
            cv2.imwrite(str(base_path_test_andrena) + "/" + str(folder) + "/" + img, image)
