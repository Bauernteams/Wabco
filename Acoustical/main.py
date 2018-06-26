# Last modified: 11.06.2018

from data_loader.sound import SoundDataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dirs import listdir
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from math import ceil

################################################################################################################################
ID = None # None für alle
useOthersAsUnknown = True

attributes = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Stoerung","Reifen","Reifendruck","Position","Fahrbahn"]

labels = [["Beton","Blaubasalt","Asphalt"]#,"Stahlbahn","Schlechtwegestrecke"]      # Belag
        ,["trocken","nass","feucht"]#,"nass/feucht]                                 # Witterung
        ,["80 km/h","50 km/h","30 km/h","40 km/h"]#, "0 km/h", '20 km/h', 'x km/h', # Geschwindigkeit
            # '80 - 0 km/h', "0 - 80 km/h",'50 - 0 km/h', '40 - 0 km/h']         
        ,['PCB - Kein', 'PCB - Puschel','PCB - Kondom']#]                           # Mikrofon
        ,None#['keine', 'LKW/Sattelzug parallel', 'Reisszwecke im Profil',          # Stoerung
            # 'CAN aus', 'Beregnung an']
        ,None#['Goodyear', 'Michelin']#, 'XYZ']                                     # Reifen
        ,['8 bar', '9 bar', '6 bar']#]                                              # Reifendruck
        ,None#[1,2,3,4]                                                             # Position
        ,['Oval']#, 'ESC-Kreisel', 'Fahrdynamikflaeche']                            # Fahrbahn
        ]
class_attributes = ["Belag","Witterung"]
#class_attributes = ["Reifen","Reifendruck"]
#class_attributes = ["Mikrofon"]
identification = ["ID","frame"]
##################################################################################################################################

if __name__ ==  '__main__':
    sdl = SoundDataLoader("configs/wabco.json")
    print("Data Loaded...")

    currentDrive, path = os.path.splitdrive(os.getcwd())
    dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-1]),"Datastore","Acoustical")
    # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
    #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
    samples = sdl.getFeaturesWithLabel(attributes,labels)
    print("Attribute Filter executed...")
        
    if useOthersAsUnknown:
        ####
        ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
        notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
        #changeIDs = range(6)
        changeIDs = notUsedSamples.ID.unique()
        for attribute in attributes:
            sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
        samples = sdl.features
        print("Outfiltered samples renamed as unknown and loaded...")
        ####
    
    # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
    train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)
    print("Data equalized and splitted...")

    class_attributes = ",".join(class_attributes)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)

    clf = make_pipeline(StandardScaler(), PCA(n_components=30), SVC(decision_function_shape="ovo", probability=True))
    print("Classification-Pipeline created...")

    clf.fit(train.drop(columns=(identification+[class_attributes])).values, classes_list)
    print("Classifier trained...")

    # predict class and probability
    if ID is None:
        p_samples = test.drop(columns=(identification+[class_attributes])).values
    else:
        p_samples = test[test["ID"] == ID].drop(columns=(identification+[class_attributes])).values
    prediction = clf.predict(p_samples)
    probability = clf.predict_proba(p_samples)
    print("Test-data predicted...")
    # /predict class and probability

    # Confusion Matrix
    y_ = [class_names[int(p)] for p in prediction]
    if ID is None:
        y_true = test[class_attributes].values
    else:
        y_true = test[test.ID == ID][class_attributes].values
    cnf_matrix = confusion_matrix(y_true, y_, class_names)
    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    # /confusion matrix

    # plot prediction boxplot
    fig,axs = plt.subplots(ceil(len(class_names)/2),2)
    ai2 = 0
    for ai, cn in enumerate(class_names):
        #print(ai%2,int(ai2), cn)
        idx = np.where(y_true==cn)
        actualClassPrediction = probability[idx]
        axs[int(ai2), ai%2].boxplot(actualClassPrediction, labels = class_names)
        axs[int(ai2), ai%2].set_title(cn)
        ai2+=0.5
    plt.show()