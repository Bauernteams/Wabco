# Last modified: 11.06.2018

from data_loader.sound import SoundDataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dirs import listdir, getHighestFilenumber
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from math import ceil

################################################################################################################################
ID = [332] # None für alle
useOthersAsUnknown = False
saveClassifier = True
useClassifierID = -1

attributes = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Stoerung","Reifen","Reifendruck","Position","Fahrbahn"]

labels = [["Beton","Blaubasalt","Asphalt"]#,"Stahlbahn","Schlechtwegestrecke"]      # Belag
        ,["trocken","nass"]#,"feucht","nass/feucht]                                 # Witterung
        ,["80 km/h","50 km/h","30 km/h","40 km/h"]#, "0 km/h", '20 km/h', 'x km/h', # Geschwindigkeit
            # '80 - 0 km/h', "0 - 80 km/h",'50 - 0 km/h', '40 - 0 km/h']         
        ,['PCB - Kein','PCB - Kondom']#, 'PCB - Puschel']                           # Mikrofon
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
    print("Loading Data...")
    sdl = SoundDataLoader("configs/wabco.json")

    print("Filtering Attributes...")
    currentDrive, path = os.path.splitdrive(os.getcwd())
    dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-1]),"Datastore","Acoustical")
    # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
    #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
    samples = sdl.getFeaturesWithLabel(attributes,labels)
               
    if useOthersAsUnknown:
        print("Renaming and loading outfiltered data as unknown...")
        ####
        ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
        notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
        #changeIDs = range(6)
        changeIDs = notUsedSamples.ID.unique()
        for attribute in attributes:
            sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
        samples = sdl.features
        ####
    
    # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
    print("Equalizing data and splitting in training- and test-data...")
    if ID is None:
        train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)
    else:
        if not samples.ID.isin(ID).any():
            samples = pd.concat([samples, sdl.features[sdl.features.ID.isin(ID)]])
        train = sdl.equalize(samples, class_attributes, randomize = True, split_train_test = None)
        test = train[train.ID.isin(ID)]
        train = train.drop(test.index.values)
        
    
    class_attributes = ",".join(class_attributes)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)

    if useClassifierID is None:
        print("Creating classification-Pipeline...")
        clf = make_pipeline(StandardScaler(), PCA(n_components=30), SVC(decision_function_shape="ovo", probability=True))
    
        print("Training Classifier...")
        clf.fit(train.drop(columns=(identification+[class_attributes])).values, classes_list)
        
        if saveClassifier:
            print("Saving classifier...")
            # TODO: in eine .txt-Datei die Paramter zu dem gespeicherten Classifier einfügen
            from sklearn.externals import joblib
            joblib.dump(clf, os.path.join("classifier",str(getHighestFilenumber("classifier")+1)+".pkl"))
    else:
        from sklearn.externals import joblib
        if useClassifierID == -1:
            useClassifierID = getHighestFilenumber("classifier")
        print("Loading Classifier-ID", useClassifierID)
        clf = joblib.load(os.path.join("classifier", str(useClassifierID)+".pkl"))

    # predict class and probability
    print("Predicting Test-Data...")
    if ID is None:
        p_samples = test.drop(columns=(identification+[class_attributes])).values
    else:
        p_samples = test[test["ID"] == ID].drop(columns=(identification+[class_attributes])).values
    prediction = clf.predict(p_samples)
    probability = clf.predict_proba(p_samples)
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