from data_loader.sound import SoundDataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dirs import listdir
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
a=4

PCA_components = 30
ID = 11

if __name__ ==  '__main__':
    sdl = SoundDataLoader("configs/wabco.json")
    currentDrive, path = os.path.splitdrive(os.getcwd())
    dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-2]),"Datastore","Wabco Audio")
    attr = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Stoerung","Reifen","Reifendruck","Position","Fahrbahn"]
    lab = [["Beton","Blaubasalt","Asphalt"]#,"Stahlbahn","Schlechtwegestrecke"]         # Belag
            ,["nass","trocken"]#,"feucht","nass/feucht]                                 # Witterung
            ,["80 km/h","50 km/h","30 km/h","40 km/h", "0 km/h"]#, "0 - 80 km/h",                 # Geschwindigkeit
                # '80 - 0 km/h', '50 - 0 km/h', '40 - 0 km/h', '20 km/h', 'x km/h']         
            ,None#['PCB - Kein', 'PCB - Puschel','PCB - Kondom']                        # Mikrofon
            ,None#['keine', 'LKW/Sattelzug parallel', 'Reisszwecke im Profil',          # Stoerung
                # 'CAN aus', 'Beregnung an']
            ,['Goodyear', 'Michelin']#, 'XYZ']                                          # Reifen
            ,None#['8 bar', '9 bar', '6 bar']                                           # Reifendruck
            ,None#[1,2,3,4]                                                             # Position
            ,['Oval']#, 'ESC-Kreisel', 'Fahrdynamikflaeche']
            ]
    class_attributes = ["Belag", "Witterung"]
    #class_attributes = ["Reifen","Reifendruck"]
    #class_attributes = ["Geschwindigkeit"]
    identification = ["ID","frame"]
    
    if a == 1:
        # Sounddatei gestückelt in den Classifier füttern
        # soll quasi das Mikrofon simulieren

        #for f in listdir(dataFolder+"/processed/Features_filt"):
        #    sdl.loadFeature_csv(dataFolder+"/processed/Features_filt/"+f)
        samples = sdl.getFeaturesWithLabel(attr,lab)
        equalized = sdl.equalize(samples, class_attributes)
    
        # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
        train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)

        class_attributes = ",".join(class_attributes)
        classes_list, class_names = sdl.Attr_to_class(train,class_attributes)
       
        clf = make_pipeline(StandardScaler(), PCA(n_components=30), SVC(decision_function_shape="ovo", probability=True))
        clf.fit(train.drop(columns=(identification+[class_attributes])).values, classes_list)
        
        ###
        # Laden einer Audiodatei, zerstückeln in einzelne Teile und füttern in den Classifier
        #audio,sr = sdl.loadRawWithID(ID)
        audio,sr = sdl.loadRawWithPath("Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\test\\526_Malek_Samo.wav")
        
        features = sdl.extractFeaturesFromAudio(audio, sr = sr)
        prediction = clf.predict(features.values)
        from collections import Counter
        if len(prediction>1):
            print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction]))
        else:
            print("\nPREDICTION:\t", Counter(class_names[int(prediction)]))
     
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
    
    #if a == 3:
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
        # VERSCHOBEN NACH main.py
    
    if a == 4:
        # SVM fitting über die Trainings- und Testdaten und anschließend aufnahme und prediction über MIKROFON
    
        samples = sdl.getFeaturesWithLabel(attr,lab)

        # Vorteil von split_train_test in sdl equalize: jede ID wird abhängig von dem equalizen gesplittet.
        # Bsp.: ID 170 (Beton, nass) besteht aus 55 frames. Insgesamt gibt es von der Klasse (Beton, nass) 2784 frames.
        # Am wenigsten frames gibt es von der Klasse (Asphalt, trocken): 1874 frames.
        # Nach dem equalizen sind es also noch 55 * 1874 / 2784 = 36 frames (Zufällig ausgewählte, wenn randomize True. Sonst die ersten).
        # Bei einem split_train_test=0.7 sind es nach dem split noch 36*7/10=25 frames im train und 11 frames im test DataFrame von der ID 170.
        # Hierdurch wird verhindert, dass der Zufall beim .sample() einige IDs komplett entfernt.
        train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)

        # Skalierung und PCA-Transformation -> Lernen mit SVM
        class_attributes = ",".join(class_attributes)
        classes_list, class_names = sdl.Attr_to_class(train,class_attributes)

        clf = make_pipeline(StandardScaler(), 
                            PCA(n_components=PCA_components), 
                            SVC(decision_function_shape="ovo", probability=True))
        clf.fit(train.drop(columns=(identification+[class_attributes])).values, classes_list)

    
        """
        Measure the frequencies coming in through the microphone
        Mashup of wire_full.py from pyaudio tests and spectrum.py from Chaco examples
        """

        import pyaudio
        import numpy as np
        import scipy.signal
        import matplotlib.pyplot as plt
        import time

        CHUNK = 16384//2

        WIDTH = 2
        DTYPE = np.int16
        MAX_INT = 32768.0

        CHANNELS = 1
        RATE = 48000

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(WIDTH),
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        # initialize sample buffer
        buffer = np.zeros(CHUNK)

        #for i in np.arange(RATE / CHUNK * RECORD_SECONDS):
        from collections import Counter
        while True:
            # read audio
            string_audio_data = stream.read(CHUNK)
            audio_data = np.fromstring(string_audio_data, dtype=DTYPE)
            normalized_data = audio_data / MAX_INT

            DF = sdl.extractFeaturesFromAudio(normalized_data)
            prediction = clf.predict(DF.values)
            if len(prediction>1):
                print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction]))
            else:
                print("\nPREDICTION:\t", class_names[int(prediction)])
            prediction2 = np.mean(clf.predict_proba(DF.values), axis=0)
            print(prediction2)


        print("* done")

        stream.stop_stream()
        stream.close()

        p.terminate()

    if a == 5:
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