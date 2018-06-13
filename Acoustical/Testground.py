from data_loader.sound import SoundDataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dirs import listdir
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
a=3

sdl = SoundDataLoader("configs/wabco.json")
currentDrive, path = os.path.splitdrive(os.getcwd())
dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-2]),"Datastore","Wabco Audio")
attr = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Stoerung","Reifen","Reifendruck","Position","Fahrbahn"]
lab = [["Beton","Blaubasalt","Asphalt","Stahlbahn"]#,"Schlechtwegestrecke"]         # Belag
        ,["nass","trocken"]#,"feucht","nass/feucht]                                 # Witterung
        ,None#["80 km/h","50 km/h","30 km/h"]#,"40 km/h", "0 - 80 km/h",                 # Geschwindigkeit
            # '80 - 0 km/h', '50 - 0 km/h', '40 - 0 km/h', '20 km/h', 'x km/h']         
        ,None#['PCB - Kein', 'PCB - Puschel','PCB - Kondom']                        # Mikrofon
        ,None#['keine', 'LKW/Sattelzug parallel', 'Reisszwecke im Profil',          # Stoerung
            # 'CAN aus', 'Beregnung an']
        ,['Goodyear', 'Michelin']#, 'XYZ']                                          # Reifen
        ,None#['8 bar', '9 bar', '6 bar']                                           # Reifendruck
        ,None#[1,2,3,4]                                                             # Position
        ,None#['Oval', 'ESC-Kreisel', 'Fahrdynamikflaeche']
        ]
#class_attributes = ["Belag", "Witterung"]
#class_attributes = ["Reifen","Reifendruck"]
class_attributes = ["Geschwindigkeit"]
identification = ["ID","frame","index"]
    
if a == 1:
    PCA_components = 5
    ID = 11
    # Sounddatei gestückelt in den Classifier füttern
    # soll quasi das Mikrofon simulieren

    #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
    sdl.loadFeature_csv(dataFolder+"/processed/features_NB.csv")

    #for f in listdir(dataFolder+"/processed/Features_filt"):
    #    sdl.loadFeature_csv(dataFolder+"/processed/Features_filt/"+f)
    samples = sdl.getFeaturesWithLabel(attr,lab)
    equalized = sdl.equalize(samples, class_attributes)

    from sklearn.decomposition import PCA
    pca = PCA(PCA_components)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(equalized)
    normalized = sc.fit_transform(train.drop(columns=(identification + class_attributes)).values)
    principalComponents = pca.fit_transform(normalized)
    
    from sklearn import svm
    clf = svm.SVC(decision_function_shape="ovo", probability = True)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)
    clf.fit(principalComponents, classes_list)
        
    ###
    # Laden einer Audiodatei, zerstückeln in einzelne Teile und füttern in den Classifier
    ID = 11
    audio,sr = sdl.loadRawWithID(ID)
    features = sdl.extractFeaturesFromAudio(audio, sr = sr)
    features2 = sdl.features[sdl.features.ID == 11]
    normalized = sc.transform(features.values)
    components = pca.transform(normalized)
    prediction = clf.predict(components)
    if len(prediction>1):
        print("\nPREDICTION:\t", [class_names[int(p)] for p in prediction])
    else:
        print("\nPREDICTION:\t", class_names[int(prediction)])
     
if a == 2:
    # Aufnahme des Audios vom Mikrofon und Plotten der frequenzen
    """
    Measure the frequencies coming in through the microphone
    Mashup of wire_full.py from pyaudio tests and spectrum.py from Chaco examples
    """

    import pyaudio
    import numpy as np
    import scipy.signal
    import matplotlib.pyplot as plt
    import time

    CHUNK = 1024*2

    WIDTH = 2
    DTYPE = np.int16
    MAX_INT = 32768.0

    CHANNELS = 1
    RATE = 11025*1
    RECORD_SECONDS = 20

    j = np.complex(0,1)


    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    # initialize sample buffer
    buffer = np.zeros(CHUNK * 2)

    #for i in np.arange(RATE / CHUNK * RECORD_SECONDS):
    plt.ion()
    while True:
        # read audio
        string_audio_data = stream.read(CHUNK)
        audio_data = np.fromstring(string_audio_data, dtype=DTYPE)
        normalized_data = audio_data / MAX_INT
        freq_data = np.fft.fft(normalized_data)

        plt.plot(np.abs(freq_data))
        plt.show()
        plt.pause(0.1)
        plt.clf()

    print("* done")

    stream.stop_stream()
    stream.close()

    p.terminate()
    
if a == 3:
    ID = None # None für alle
    # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
    #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
    sdl.loadFeature_csv(dataFolder+"/processed/features_NB.csv")
    samples = sdl.getFeaturesWithLabel(attr,lab) 
    
    # Vorteil von split_train_test in sdl equalize: jede ID wird abhängig von dem equalizen gesplittet.
    # Bsp.: ID 170 (Beton, nass) besteht aus 55 frames. Insgesamt gibt es von der Klasse (Beton, nass) 2784 frames.
    # Am wenigsten frames gibt es von der Klasse (Asphalt, trocken): 1874 frames.
    # Nach dem equalizen sind es also noch 55 * 1874 / 2784 = 36 frames (Zufällig ausgewählte, wenn randomize True. Sonst die ersten).
    # Bei einem split_train_test=0.7 sind es nach dem split noch 36*7/10=25 frames im train und 11 frames im test DataFrame von der ID 170.
    # Hierdurch wird verhindert, dass der Zufall beim .sample() einige IDs komplett entfernt.
    train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)

    # Skalierung und PCA-Transformation
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    normalized = sc.fit_transform(train.drop(columns=(identification+class_attributes)).values)
    from sklearn.decomposition import PCA
    pca = PCA(30)
    principalComponents = pca.fit_transform(normalized)
    print(pca.components_)


    # Lernen mit SVM
    from sklearn import svm
    clf = svm.SVC(decision_function_shape="ovo", probability=True)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)
    clf.fit(principalComponents, classes_list)
        
    # predict class
    if ID is None:
        x_  = sc.transform(test.drop(columns=(identification + class_attributes)).values)
    else:
        x_  = sc.transform(test[test["ID"] == ID].drop(columns=(identification + class_attributes)).values)
    pc_ = pca.transform(x_)
    
    prediction = clf.predict(pc_)
    # /predict class

    # predict probability
    if ID is None:
        x2_  = sc.transform(test.drop(columns=(identification + class_attributes)).values)
    else:
        x2_  = sc.transform(test[test["ID"] == ID].drop(columns=(identification + class_attributes)).values)
    pc2_ = pca.transform(x2_)
    prediction2 = clf.predict_proba(pc2_)
    #print(prediction2)
    # /predict probablity

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    test["class"] = sdl.combineAttributes(test, class_attributes)
    newcn = np.array([",".join(cn) for cn in class_names])
    
    if isinstance(class_attributes, list) and len(class_attributes)>1:
        y_ = [",".join(class_names[int(p)]) for p in prediction]
        test["class"] = sdl.combineAttributes(test, class_attributes)
        newcn = np.array([",".join(cn) for cn in class_names])
    else: 
        y_ = [class_names[int(p)] for p in prediction]
        test["class"] = test[class_attributes]
        newcn = class_names

    if ID is None:
        y_true = test["class"].values
    else:
        y_true = test[test.ID == ID]["class"].values
    
    cnf_matrix = confusion_matrix(y_true, y_, newcn)
    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=newcn,
                          title='Confusion matrix, without normalization')

    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=newcn, normalize=True,
                          title='Normalized confusion matrix')

    # /confusion matrix

    # plot prediction boxplot
    fig,axs = plt.subplots(len(newcn)//2,2)
    ai2 = 0
    for ai, cn in enumerate(newcn):
        #print(ai%2,int(ai2), cn)
        idx = np.where(y_true==cn)
        actualClassPrediction = prediction2[idx]
        axs[int(ai2), ai%2].boxplot(actualClassPrediction, labels = newcn)
        axs[int(ai2), ai%2].set_title(cn)
        ai2+=0.5
    plt.show()
    
if a == 4:
    # SVM fitting über die Trainings- und Testdaten und anschließend aufnahme und prediction über MIKROFON
    
    #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
    sdl.loadFeature_csv(dataFolder+"/processed/features_NB.csv")
    samples = sdl.getFeaturesWithLabel(attr,lab)
    
    # Vorteil von split_train_test in sdl equalize: jede ID wird abhängig von dem equalizen gesplittet.
    # Bsp.: ID 170 (Beton, nass) besteht aus 55 frames. Insgesamt gibt es von der Klasse (Beton, nass) 2784 frames.
    # Am wenigsten frames gibt es von der Klasse (Asphalt, trocken): 1874 frames.
    # Nach dem equalizen sind es also noch 55 * 1874 / 2784 = 36 frames (Zufällig ausgewählte, wenn randomize True. Sonst die ersten).
    # Bei einem split_train_test=0.7 sind es nach dem split noch 36*7/10=25 frames im train und 11 frames im test DataFrame von der ID 170.
    # Hierdurch wird verhindert, dass der Zufall beim .sample() einige IDs komplett entfernt.
    train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)

    # Skalierung und PCA-Transformation
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    normalized = sc.fit_transform(train.drop(columns=(identification+class_attributes)).values)
    from sklearn.decomposition import PCA
    pca = PCA(5)
    principalComponents = pca.fit_transform(normalized)

    # Lernen mit SVM
    from sklearn import svm
    clf = svm.SVC(decision_function_shape="ovo", probability=True)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)
    clf.fit(principalComponents, classes_list)

    
    """
    Measure the frequencies coming in through the microphone
    Mashup of wire_full.py from pyaudio tests and spectrum.py from Chaco examples
    """

    import pyaudio
    import numpy as np
    import scipy.signal
    import matplotlib.pyplot as plt
    import time

    CHUNK = 16384

    WIDTH = 2
    DTYPE = np.int16
    MAX_INT = 32768.0

    CHANNELS = 1
    RATE = 48000
    RECORD_SECONDS = 20

    j = np.complex(0,1)


    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    # initialize sample buffer
    buffer = np.zeros(CHUNK * 2)

    #for i in np.arange(RATE / CHUNK * RECORD_SECONDS):
    plt.ion()
    while True:
        # read audio
        string_audio_data = stream.read(CHUNK)
        audio_data = np.fromstring(string_audio_data, dtype=DTYPE)
        normalized_data = audio_data / MAX_INT

        DF = sdl.extractFeaturesFromAudio(normalized_data)
        x_  = sc.transform(DF.values)
        pc_ = pca.transform(x_)
        prediction = clf.predict(pc_)
        if len(prediction>1):
            print("\nPREDICTION:\t", [class_names[int(p)] for p in prediction])
        else:
            print("\nPREDICTION:\t", class_names[int(prediction)])
        prediction2 = np.mean(clf.predict_proba(pc_), axis=0)
        print(prediction2)


    print("* done")

    stream.stop_stream()
    stream.close()

    p.terminate()

if a == 5:
    PCA_components = 5
    ID = 22
    # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten, anschließende einzelne Klassifizierung
    # von neu geladenen Audiodaten über die ID
    sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")

    samples = sdl.getFeaturesWithLabel(attr,lab)
    equalized = sdl.equalize(samples, class_attributes)

    from sklearn.decomposition import PCA
    pca = PCA(PCA_components)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    from sklearn.model_selection import train_test_split
    train, _ = train_test_split(equalized)
    normalized = sc.fit_transform(train.drop(columns=(identification + class_attributes)).values)
    principalComponents = pca.fit_transform(normalized)
    #principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
    #principalDf = pd.DataFrame(data = principalComponents, columns = ["principal component "+str(i) for i in principalComponents.shape[1]])
    #finalDf = pd.concat([principalDf, train[['Belag',"Witterung"]]], axis = 1)
    
    # SVM 3 - skalierte und PCA getrennte Trainingsdaten 
    # PCA(2) Counter({True: 716, False: 373})
    # PCA(5) Counter({True: 1009, False: 80})
    from sklearn import svm
    clf = svm.SVC(decision_function_shape="ovo", probability=True)
    classes_list, class_names = sdl.Attr_to_class(train,class_attributes)
    clf.fit(principalComponents, classes_list)

    # predict class
    audio,sr = sdl.loadRawWithID(ID)
    test = sdl.extractFeaturesFromAudio(audio)
    x_  = sc.transform(test.values)
    pc_ = pca.transform(x_)
    prediction = clf.predict(pc_)
    # /predict class

    # predict probability
    x2_  = sc.transform(test.values)
    pc2_ = pca.transform(x2_)
    prediction2 = clf.predict_proba(pc2_)
    # /predict probablity

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    ID_attr = sdl.attributes[class_attributes][sdl.attributes.ID == ID].values[0]
    for ca,ID_att in zip(class_attributes, ID_attr):
        test[ca] = ID_att
    test["class"] = sdl.combineAttributes(test, class_attributes)
    newcn = np.array([",".join(cn) for cn in class_names])
    cnf_matrix = confusion_matrix(test["class"].values, [",".join(class_names[int(p)]) for p in prediction], newcn)
    print("Mean of confidence:\n", "\n".join(["\t".join([n,str(v)]) for n,v in zip(newcn,np.mean(prediction2,axis=0))]))
    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=newcn,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    sdl.plot_confusion_matrix(cnf_matrix, classes=newcn, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

if a == 6:
    # Neues Featuresset-File erstellen mit "meinen" features
    import librosa

    n_fft = 16384
    from utils.dirs import listdir
    files = listdir(os.path.join(dataFolder,"raw","1_set"))
    #file = "11_Run005.Mik2.wav" # ersetzen durch for loop über alle files
    
    all_features = pd.DataFrame()
    for file in files:
        ID = int(file.split("_")[0])
        print(ID)
        audio, sr = librosa.load(os.path.join(dataFolder,"raw","1_set",file))
        features = sdl.extractFeaturesFromAudio(audio,n_fft=n_fft, sr=sr)
        features["ID"] = ID
        all_features = pd.concat((all_features,features), ignore_index=True, copy=False)

    all_features.to_csv("experiments\\Wabco\\data\\processed\\features_NB.csv",";")
    
if a == 7:
    t = pd.DataFrame()
    t["B"] = sdl.getAttributeToFeatures(samples, "Belag")
    t["W"] = sdl.getAttributeToFeatures(samples, "Witterung")
    t["G"] = sdl.getAttributeToFeatures(samples, "Geschwindigkeit")
    t["R"] = sdl.getAttributeToFeatures(samples, "Reifen")
    t["Rd"] = sdl.getAttributeToFeatures(samples, "Reifendruck")
    t["P"] = sdl.getAttributeToFeatures(samples, "Position")
    t["M"] = sdl.getAttributeToFeatures(samples, "Mikrofon")

    sdl.combineAttributes(t, ["B","W"]).value_counts()
    t["B,W"] = sdl.combineAttributes(t, ["B","W"])
    t["B,W,G"] = sdl.combineAttributes(t, ["B,W","G"])

    # Nützliche Pandas Befehle
    ## Überprüfung, welche IDs falsch klassifiziert wurden und womit:
    test["correct"] = [a == b for a,b in zip(list(y_true), y_)]
    test["prediction"] = y_
    test[["ID", class_attributes, "prediction"]][test.correct == False]
    ##
    