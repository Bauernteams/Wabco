from data_loader.sound import SoundDataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dirs import listdir, getHighestFilenumber
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
a=15

################################################################################################################################
Run = list(range(11,16,1)) # None für alle
useOthersAsUnknown = False
saveClassifier = False
useClassifierID = None

if __name__ ==  '__main__':
    sdl = SoundDataLoader("configs/wabco.json")
    currentDrive, path_in = os.path.splitdrive(os.getcwd())
    dataFolder = os.path.join(currentDrive,os.path.sep.join(path_in.split(os.path.sep)[:-2]),"Datastore","Wabco Audio")
    attributes = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Reifen","Reifendruck","Position","Fahrbahn"]
    labels = [["Beton","Blaubasalt","Asphalt","Stahlbahn"]#,"Schlechtwegestrecke"]         # Belag
            ,["nass","trocken"]#,"feucht","nass/feucht]                                 # Witterung
            ,["80 km/h","50 km/h","30 km/h","40 km/h", "0 km/h"]#, "0 - 80 km/h",                 # Geschwindigkeit
                # '80 - 0 km/h', '50 - 0 km/h', '40 - 0 km/h', '20 km/h', 'x km/h']         
            ,None#['PCB - Kein', 'PCB - Puschel','PCB - Kondom']                        # Mikrofon
            ,None#['Goodyear', 'Michelin']#, 'XYZ']                                          # Reifen
            ,None#['8 bar', '9 bar', '6 bar']#  ]                                         # Reifendruck
            ,None#[1,2,3,4]                                                             # Position
            ,['Oval']#, 'ESC-Kreisel', 'Fahrdynamikflaeche']
            ]
    
    unknownAttributes   = ["Geschwindigkeit"]   # Attribute, in denen sich Unknown Labels befinden
    unknownLabels       = ["0 km/h"]            # Diese Labels werden als Unknown klassifiziert
    class_attributes = ["Belag", "Witterung"]
    #class_attributes = ["Reifen","Reifendruck"]
    #class_attributes = ["Geschwindigkeit"]
    identification = ["ID","frame"]
    
    if a == 1:
        # Sounddatei gestückelt in den Classifier füttern
        # soll quasi das Mikrofon simulieren

        #for f in listdir(dataFolder+"/processed/Features_filt"):
        #    sdl.loadFeature_csv(dataFolder+"/processed/Features_filt/"+f)
        samples = sdl.getFeaturesWithLabel(attributes,labels)
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
        audio,sr = sdl.loadRawWithPath("Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\180723-24\\Phillips\\533_Run259.Mik3.wav")
        
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
    
    if a == 3:
        # SVR (Regression) der Belagklassen, die durch einen Reibwert (0<R<1) ersetzt wurden.
        
        from sklearn.svm import SVR        
        print("Loading Data...")
        sdl = SoundDataLoader("configs/wabco.json")

        print("Filtering Attributes...")
        currentDrive, path_in = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path_in.split(os.path.sep)[:-1]),"Datastore","Acoustical")
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
        #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
        samples = sdl.getFeaturesWithLabel(attributes,labels)

        ### Umwandlung Belag+Witterung in Reibwert:
        # Platzhalter-Werte, richtige Werte müssen noch gefunden werden
        trans = dict({"Asphalt"             : 0.9, 
                      "Beton"               : 0.8, 
                      "Blaubasalt"          : 0.5, 
                      "Stahlbahn"           : 0.2,
                      "Schlechtwegestrecke" : 0})

        trans2 = dict({"trocken" : 1,
                       "nass" : 2,
                       "feucht" : 1,
                       "nass/feucht": 1})
        
        sdl.attributes["Reibwert"] = sdl.attributes.apply(lambda b: trans[b["Belag"]]/trans2[b["Witterung"]], axis=1)
        class_attributes = ["Reibwert"]
        ###
        
        
        ### Nicht verwendete Soundfiles benutzen als "Unbekannt":
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
        ###
    
        # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
        print("Equalizing data and splitting in training- and test-data...")
        if Run is None:
            train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)
        else:
            if not samples.ID.isin(Run).any():
                samples = pd.concat([samples, sdl.features[sdl.features.ID.isin(Run)]])
            train = sdl.equalize(samples, class_attributes, randomize = True, split_train_test = None)
            test = train[train.ID.isin(Run)]
            train = train.drop(test.index.values)
        
    
        class_attributes = ",".join(class_attributes)
        classes_list, class_names = sdl.Attr_to_class(train,class_attributes)

        if useClassifierID is None:
            print("Creating classification-Pipeline...")
            clf = make_pipeline(StandardScaler(), PCA(n_components=30), SVR())    
            print("Training Classifier...")
            Y = train.ID.apply(lambda id: sdl.attributes.Reibwert[sdl.attributes.ID == id].values)
            X = train.drop(columns=(identification+[class_attributes]))
            #clf.fit(train.drop(columns=(identification+[class_attributes])).values, classes_list)
            clf.fit(X, Y)
        
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
        if Run is None:
            p_samples = test.drop(columns=(identification+[class_attributes])).values
        else:
            p_samples = test[test["ID"] == Run].drop(columns=(identification+[class_attributes])).values
        prediction = clf.predict(p_samples)
        # /predict class and probability

        # plot prediction boxplot
        y_ = prediction
        if Run is None:
            y_true = test[class_attributes].values
        else:
            y_true = test[test.ID == Run][class_attributes].values
        from math import ceil
        lab = test[class_attributes].unique()
        if len(lab)==1:
            plt.title(["Reibwert:", str(lab)])
            plt.axis([0.8,1.2,0,1])
            plt.axhline(y=y_true[0])
            plt.boxplot(prediction)
        else:
            fig,axs = plt.subplots(ceil(len(lab)/2),2, )
            ai2 = 0
            for ai, cn in enumerate(test[class_attributes].unique()):
                #print(ai%2,int(ai2), cn)
                idx = np.where(y_true==cn)
                actualClassPrediction = prediction[idx]
                axs[int(ai2), ai%2].boxplot(actualClassPrediction)
                axs[int(ai2), ai%2].set_title(["Reibwert:" + str(cn)])
                axs[int(ai2), ai%2].axis([0.8,1.2,0,1])
                axs[int(ai2), ai%2].axhline(y=cn)
                ai2+=0.5
        plt.show()

    if a == 4:    
        """
        Measure the frequencies coming in through the microphone
        Mashup of wire_full.py from pyaudio tests and spectrum.py from Chaco examples
        """
        # SVM fitting über die Trainings- und Testdaten und anschließend aufnahme und prediction über MIKROFON
    
        samples = sdl.getFeaturesWithLabel(attributes,labels)
                       
        if useOthersAsUnknown:
            print("Renaming and loading outfiltered data as unknown...")
            ####
            ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
            #notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
            asUnknownSamples = sdl.getFeaturesWithLabel(unknownAttributes,unknownLabels)
            changeIDs = asUnknownSamples.ID.unique()
            for attribute in attributes:
                sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
            samples = pd.concat([samples, asUnknownSamples])
            ####

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

        from sklearn.externals import joblib
        useClassifierID = getHighestFilenumber("classifier")
        print("Loading Classifier-ID", useClassifierID)
        clf = joblib.load(os.path.join("classifier", str(useClassifierID)+".pkl"))



        import pyaudio
        import numpy as np
        import scipy.signal
        import matplotlib.pyplot as plt
        import time

        CHUNK = 16384*5

        WIDTH = 2
        DTYPE = np.int16
        MAX_INT = 32768.0
        FORMAT = pyaudio.paInt16

        CHANNELS = 1
        RATE = 48000
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = "file_RtCh.wav"

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

        # initialize sample buffer
        buffer = np.zeros(CHUNK)

        #for i in np.arange(RATE / CHUNK * RECORD_SECONDS):
        from collections import Counter
        import time
        while True:
            # read audio
            string_audio_data = stream.read(CHUNK)
            audio_data = np.fromstring(string_audio_data, dtype=DTYPE)
            normalized_data = audio_data / MAX_INT

            DF = sdl.extractFeaturesFromAudio(normalized_data, hop_length = None)
            prediction = clf.predict(DF.values)
            if len(prediction>1):
                print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction]))
            else:
                print("\nPREDICTION:\t", class_names[int(prediction)])
            prediction2 = np.mean(clf.predict_proba(DF.values), axis=0)
            print(prediction2)
            time.sleep(1)


        print("* done")

        stream.stop_stream()
        stream.close()

        p.terminate()

    if a == 5:
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten, anschließende einzelne Klassifizierung
        # von neu geladenen Audiodaten über die ID
        print("Loading Data...")
        sdl = SoundDataLoader("configs/wabco.json")

        print("Filtering Attributes...")
        currentDrive, path_in = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path_in.split(os.path.sep)[:-1]),"Datastore","Acoustical")
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
        #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
        samples = sdl.getFeaturesWithLabel(attributes,labels)
               
        if useOthersAsUnknown:
            print("Renaming and loading outfiltered data as unknown...")
            ####
            ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
            #notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
            asUnknownSamples = sdl.getFeaturesWithLabel(unknownAttributes,unknownLabels)
            changeIDs = asUnknownSamples.ID.unique()
            for attribute in attributes:
                sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
            samples = pd.concat([samples, asUnknownSamples])
            ####
    
        # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
        print("Equalizing data and splitting in training- and test-data...")
        train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)
        
    
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

        # predict class
        audio,sr = sdl.loadRawWithID(Run)
        test = sdl.extractFeaturesFromAudio(audio)
        prediction = clf.predict(test.values)
        # /predict class

        # predict probability
        prediction2 = clf.predict_proba(test.values)
        # /predict probablity

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        ID_attr = sdl.attributes[class_attributes][sdl.attributes.ID == Run].values[0]
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
            Run = int(file.split("_")[0])
            print(Run)
            audio, sr = librosa.load(os.path.join(dataFolder,"raw","1_set",file))
            features = sdl.extractFeaturesFromAudio(audio,n_fft=n_fft, sr=sr)
            features["ID"] = Run
            all_features = pd.concat((all_features,features), ignore_index=True, copy=False)

        all_features.to_csv("experiments\\Wabco\\data\\processed\\features_NB.csv",";")
    
    if a == 7:
        test = pd.DataFrame()
        test["B"] = sdl.getAttributeToFeatures(samples, "Belag")
        test["W"] = sdl.getAttributeToFeatures(samples, "Witterung")
        test["G"] = sdl.getAttributeToFeatures(samples, "Geschwindigkeit")
        test["R"] = sdl.getAttributeToFeatures(samples, "Reifen")
        test["Rd"] = sdl.getAttributeToFeatures(samples, "Reifendruck")
        test["P"] = sdl.getAttributeToFeatures(samples, "Position")
        test["M"] = sdl.getAttributeToFeatures(samples, "Mikrofon")
        test["ID"] = sdl.getAttributeToFeatures(samples, "ID")

        sdl.combineAttributes(test, ["B","W"]).value_counts()
        test["B,W"] = sdl.combineAttributes(test, ["B","W"])
        test["B,W,G"] = sdl.combineAttributes(test, ["B,W","G"])
        test["B,W,G"].value_counts()

        ### Ausgabe der Frame-Nummern und ID der falsch klassifizierten samples:
        class_attributes = ["B,W"]
        y_ = clf.predict(samples.drop(columns=(["Belag","Witterung","Belag,Witterung","ID","frame"])))      
        test["prediction"] = [class_names[int(y__)] for y__ in y_]
        test = test.sort_values("ID")
        falses = test[class_attributes] != test["prediction"]
        test["frame_length"] = test.ID.apply(lambda id: len(sdl.features[sdl.features.ID == id]))
        test[["ID","frame","frame_length"]][falses].values[:]

        ###
        import seaborn as sns
        sns.distplot(test.apply(lambda row: row["frame"]/row["frame_length"], axis=1))
        plt.xlabel("frame-number / frame_length")
        plt.title("Distribitution of false classification over frame position")
        plt.show()
        ### 

    if a == 8:
        # Ausgabe der Frame-Nummern der falsch klassifizierten samples um zu kontrollieren, ob es auffälligkeiten gibt:

        print("Loading Data...")
        sdl = SoundDataLoader("configs/wabco.json")

        print("Filtering Attributes...")
        currentDrive, path_in = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path_in.split(os.path.sep)[:-1]),"Datastore","Acoustical")
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
        #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
        samples = sdl.getFeaturesWithLabel(attributes,labels)
               
        if useOthersAsUnknown:
            print("Renaming and loading outfiltered data as unknown...")
            ####
            ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
            #notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
            asUnknownSamples = sdl.getFeaturesWithLabel(unknownAttributes,unknownLabels)
            changeIDs = asUnknownSamples.ID.unique()
            for attribute in attributes:
                sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
            samples = pd.concat([samples, asUnknownSamples])
            ####
    
        # Ausgleichen der Anzahl an samples der jeweiligen Klasse und aufteilen in Trainings- und Testdatensätze
        print("Equalizing data and splitting in training- and test-data...")
        if Run is None:
            train, test = sdl.equalize(samples, class_attributes, randomize = True, split_train_test=0.7)
        else:
            if not samples.ID.isin(Run).any():
                samples = pd.concat([samples, sdl.features[sdl.features.ID.isin(Run)]])
            train = sdl.equalize(samples, class_attributes, randomize = True, split_train_test = None)
            test = train[train.ID.isin(Run)]
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
        if Run is None:
            p_samples = test.drop(columns=(identification+[class_attributes])).values
        else:
            p_samples = test[test["ID"] == Run].drop(columns=(identification+[class_attributes])).values
        prediction = clf.predict(p_samples)
        probability = clf.predict_proba(p_samples)
        # /predict class and probability

        # Confusion Matrix
        y_ = [class_names[int(p)] for p in prediction]
        if Run is None:
            y_true = test[class_attributes].values
        else:
            y_true = test[test.ID == Run][class_attributes].values
        cnf_matrix = confusion_matrix(y_true, y_, class_names)
        plt.figure()
        sdl.plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
        plt.figure()
        sdl.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
        # /confusion matrix

        # plot prediction boxplot
        from math import ceil
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

    if a == 9:
        # Aufnahme mit dem Mikrofon und abspeichern
        import pyaudio
        import wave
 
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 96000    
        CHUNK = 16384
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = "file_RtCh.wav"
 
        audio = pyaudio.PyAudio()
 
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print("recording...")
        frames = []
 
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")
 
 
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
 
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    if a == 10:
        
        import pyaudio
        import numpy as np
        import scipy.signal
        import matplotlib.pyplot as plt
        import time
        import wave
        import os

        
        #######################        TestID & Unterordner erstellen         #######################
        TEST_ID = input("Bitte TestIDs eingeben: ")
        #folder = str(testID) + "/" + str(testIDRound) 
        folder = "Q:/Repositories/Wabco/Datastore/Acoustical/180723-24/Phillips/"

        CHUNK = 16384

        WIDTH = 2
        DTYPE = np.int16
        MAX_INT = 32768.0
        FORMAT = pyaudio.paInt16

        CHANNELS = 1
        RATE = 48000
        WAVE_OUTPUT_FILENAME = TEST_ID

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

        print("* recording")
        # initialize sample buffer
        buffer = np.zeros(CHUNK)

        allData = []
        try:
            while True:
                string_audio_data = stream.read(CHUNK)
                allData.append(string_audio_data)
        except:
            pass
        print("finished recording") 
        # stop Recording
        stream.stop_stream()
        stream.close()
        p.terminate()
 
        waveFile = wave.open(os.path.join(folder,WAVE_OUTPUT_FILENAME), 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(p.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(allData))
        waveFile.close()

    if a == 11:
        # Ändern der Namen der neuen Aufnahmen (ID hinzufügen)

        import os
        from shutil import copyfile
        
        from data_loader.sound import SoundDataLoader
        Mikro = {"Mik1":["PCB - Kein", "PCB - Puschel", "PCB - Kondom"], "Mik2":["PCB - Kein", "PCB - Puschel", "PCB - Kondom"], "Mik3":["Phillips"]}
        MikrofonPosition = {"Mik1":[1,3,7], "Mik2":[2,4,5,6], "Mik3":[1]}
        
        sdl = SoundDataLoader("configs/wabco.json")
        attributes = sdl.attributes
        
        path_in = "Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\180723-24\\PCB_o\\"
        path_out = "Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\180723-24\\test\\"
        files = os.listdir(path_in)
        i = 0
        for f in files:
            f_split = f.split(".")
            Run = f_split[0][3:]
            if "_" in Run:
                i+=1
                Run_no = Run.split("_")[0]
            else:
                Run_no = Run
                i=0

            Mik = f_split[1]
            #if "_" in Run:
                #continue
            copyfile(os.path.join(path_in,f), os.path.join(path_out, str(sdl.attributes.ID[(sdl.attributes.Nr == int(Run_no)) & 
                                                                                           (sdl.attributes.Mikrofon.isin(Mikro[Mik])) & 
                                                                                           (sdl.attributes.Position.isin(MikrofonPosition[Mik]))].values[i]) + "_Run" + str(Run) + "." + str(Mik) + ".wav"))
            i=0

    if a == 13:
        #Umwandeln von "coma-sperated-value".csv-Datei in "semi-coma-seperated-value".csv-Datei

        path_in = "Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\t.csv"
        path_out = "Q:\\Repositories\\Wabco\\Datastore\\Acoustical\\attributes.csv"
        import pandas as pd
        df = pd.DataFrame()
        df = pd.read_csv(path_in)
        df.to_csv(path_out,";")

    if a == 14:
        # Normales Ausführen der Klassifizierung mit anschließendem plotten der Hauptkomponentenanalyse (SVM Darstellung)

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
        ID = [10,16,22,28] # None für alle
        useOthersAsUnknown = False
        saveClassifier = True
        useClassifierID = None
        pca_n_components = 30

        attributes = ["Belag", "Witterung","Geschwindigkeit","Mikrofon","Stoerung","Reifen","Reifendruck","Position","Fahrbahn"]

        labels = [["Beton","Blaubasalt","Asphalt","Stahlbahn"]#,"Schlechtwegestrecke"]      # Belag
                ,["nass","trocken"]#,"feucht","nass/feucht]                                 # Witterung
                ,["80 km/h","50 km/h","30 km/h"]#,"40 km/h", "0 km/h", '20 km/h', 'x km/h', # Geschwindigkeit
                    # '80 - 0 km/h', "0 - 80 km/h",'50 - 0 km/h', '40 - 0 km/h']         
                ,['PCB - Kein']#,'PCB - Kondom', 'PCB - Puschel', "Phillips"]#                     # Mikrofon
                ,['keine']#, 'LKW/Sattelzug parallel', 'Reisszwecke im Profil',          # Stoerung
                    # 'CAN aus', 'Beregnung an']
                ,None#['Goodyear', 'Michelin']#, 'XYZ']                                     # Reifen
                ,None#['8 bar', '9 bar', '6 bar']#]                                         # Reifendruck
                ,None#[1,2,3,4]                                                             # Position
                ,['Oval']#, 'ESC-Kreisel', 'Fahrdynamikflaeche']                            # Fahrbahn
                ]

        unknownAttributes   = ["Geschwindigkeit"]   # Attribute, in denen sich Unknown Labels befinden
        unknownLabels       = ["0 km/h"]            # Diese Labels werden als Unknown klassifiziert

        dropIDs = None#[546]#None

        class_attributes = ["Belag"]#,"Witterung"]
        #class_attributes = ["Reifen","Reifendruck"]
        #class_attributes = ["Mikrofon"]
        identification = ["ID","frame"]
        ##################################################################################################################################

        print("Loading Data...")
        sdl = SoundDataLoader("configs/wabco.json")

        print("Filtering Attributes...")
        currentDrive, path = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-1]),"Datastore","Acoustical")
        # SVM fitting und prediction nach Aufteilung der csv-feature tabelle in trainings und testdaten
        #sdl.loadFeature_csv(dataFolder+"/processed/librosaFeatures.csv")
        samples = sdl.getFeaturesWithLabel(attributes,labels)

        if dropIDs is not None:
            samples = samples[samples.ID != dropIDs]
               
        if useOthersAsUnknown:
            print("Renaming and loading outfiltered data as unknown...")
            ####
            ## Nicht verwendete Soundfiles benutzen als "Unbekannt":
            #notUsedSamples = pd.concat([sdl.features,samples]).drop_duplicates(keep=False)
            asUnknownSamples = sdl.getFeaturesWithLabel(unknownAttributes,unknownLabels)
            changeIDs = asUnknownSamples.ID.unique()
            for attribute in attributes:
                sdl.changeLabel_ofIDs(changeIDs, attribute, "unknown-"+attribute)
            samples = pd.concat([samples, asUnknownSamples])
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
            clf = make_pipeline(StandardScaler(), PCA(n_components=pca_n_components), SVC(decision_function_shape="ovo", probability=True))
    
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
            p_samples = test[test.ID.isin(ID)].drop(columns=(identification+[class_attributes])).values
        prediction = clf.predict(p_samples)
        probability = clf.predict_proba(p_samples)

        from sklearn.decomposition import PCA
        from sklearn.datasets import load_iris
        from sklearn import svm
        from sklearn import cross_validation
        import pylab as pl
        import numpy as np
        from collections import Counter

        # mit 3d
        y_true = test[class_attributes].values
        scaled = clf.named_steps["standardscaler"].transform(p_samples)
        pca_transformed = clf.named_steps["pca"].transform(scaled)
        for ii in range(0, pca_transformed.shape[1] - 1):
            #for i in range(0, pca_transformed.shape[0]):
            #    if prediction[i] == 0:
            #     c1 = pl.scatter(pca_transformed[i,0+ii],pca_transformed[i,1+ii],c='r',    s=30,marker='+')
            #    elif prediction[i] == 1:
            #     c2 = pl.scatter(pca_transformed[i,0+ii],pca_transformed[i,1+ii],c='g',    s=30,marker='o')
            
            
            colors = ["r","g","b","k","y","c","m"]
            markers = ["+","o","^", ">", "<", "v", "x", "p"]
            i=0
            for ai, cn in enumerate(class_names):
                #print(ai%2,int(ai2), cn)
                idxs = [i for i, x in enumerate(y_true) if x == cn]
                for idx in idxs:
                    pl.scatter(pca_transformed[idx, 0+ii], pca_transformed[idx, 1+ii], c=colors[i], s=30, marker=markers[i])
                i+=1
                pl.legend(cn)

            x_min, x_max = pca_transformed[:, 0+ii].min() - 1,   pca_transformed[:,0+ii].max() + 1
            y_min, y_max = pca_transformed[:, 1+ii].min() - 1,   pca_transformed[:, 1+ii].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),   np.arange(y_min, y_max, .1))

            clfList = [None]*pca_transformed.shape[1]
            for iii in range(0, pca_transformed.shape[1]):
                if iii == ii: 
                    clfList[iii] = xx.ravel()
                elif iii == ii+1:
                    clfList[iii] = yy.ravel()
                else:
                    clfList[iii] = np.zeros(xx.ravel().shape[0])
            
            temp = list(zip(*clfList))
            Z = clf.named_steps["svc"].predict(temp)
            pred_class = [class_names[int(p)] for p in Z]
            Z = Z.reshape(xx.shape)
            pl.contour(xx, yy, Z, color=colors)
            pl.title(Counter(pred_class))
            plt.grid()
            pl.show()

    if a == 15:
        #Auswahl einer Audio-Datei die geladen und klassifiziert wird
        
        
        # get the classifier
        from sklearn.externals import joblib
        import json
        from utils.dirs import getHighestFilenumber
        import os
        from data_loader.sound import SoundDataLoader
        import time
        import numpy as np

        
        useClassifierID = -1
        
        sdl = SoundDataLoader("configs/wabco.json")

        if useClassifierID == -1:
            useClassifierID = getHighestFilenumber("classifier")
        print("Loading Classifier-ID", useClassifierID)
        clf = joblib.load(os.path.join("classifier", str(useClassifierID)+".pkl"))
        with open(os.path.join("classifier", str(useClassifierID)+".json")) as f:
            clf_info = json.load(f)
            class_names = clf_info["class_names"]

        # load the audio data
        #ID = int(input("Please choose an ID to be classified:"))
        ID = 11
        audio, sr = sdl.loadRawWithID(ID)

        # zerstückeln des Audios:
        CHUNK = 16384
        prediction2 = []
        for a_chunk in [audio[i*CHUNK:(i+1)*CHUNK] for i in range(len(audio)//CHUNK)]: 
            features = sdl.extractFeaturesFromAudio(np.array(a_chunk), sr = sr, drop_beg=2, drop_end=-2, hop_length = None)#CHUNK)

            for f in features.values:
                prediction = clf.predict(f.reshape(1,-1))
                prob = clf.predict_proba(f.reshape(1,-1))
                prediction2.append(prediction)
                from collections import Counter
                if len(prediction>1):
                    if prob[0,int(prediction)] > 0.6:
                        print("\nPREDICTION:\t", class_names[int(prediction)], "with",  prob[0,int(prediction)].round(2), "%")
                    else:
                        print("\n PREDICTION not sure, maximum is:\t", class_names[list(prob[0]).index(max(prob[0]))], "with", prob[0,list(prob[0]).index(max(prob[0]))].round(2), "%")
                else:
                    print("\nPREDICTION:\t", [[class_names[int(p)], prob[0,int(p)]] for p in prediction])
                time.sleep(0.5)
        print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction2]))

    if a == 16:
        # Auswahl von Frames (jeden samples), die klassifiziert werden sollen
        FRAME_ID = [0]

        test = sdl.features[sdl.features.frame.isin(FRAME_ID)]
        test["Belag"] = sdl.getAttributeToFeatures(test,"Belag")
        test["Witterung"] = sdl.getAttributeToFeatures(test,"Witterung")
        test["Belag,Witterung"] = sdl.combineAttributes(test, ["Belag","Witterung"])
        t = test.drop(columns=["ID","frame","Belag","Witterung","Belag,Witterung"])
        y_ = clf.predict(t)
        y_ = [class_names[k] for k in y_]
        y = test["Belag,Witterung"].values
        cnf_matrix = confusion_matrix(y, y_, class_names)
        sdl.plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
        plt.show()

    # GUI for the tech demo
    if a == 17:
        import tkinter as tk
        from sklearn.externals import joblib
        import json
        from utils.dirs import getHighestFilenumber
        import os
        from data_loader.sound import SoundDataLoader
        import time
        import numpy as np

        ### Classifier laden
        useClassifierID = -1
        
        sdl = SoundDataLoader("configs/wabco.json")

        if useClassifierID == -1:
            useClassifierID = getHighestFilenumber("classifier")
        print("Loading Classifier-ID", useClassifierID)
        clf = joblib.load(os.path.join("classifier", str(useClassifierID)+".pkl"))
        with open(os.path.join("classifier", str(useClassifierID)+".json")) as f:
            clf_info = json.load(f)
            class_names = clf_info["class_names"]

        # Auswertung von Audiodaten
        def startAsphalt(event):
            ID = 11 # Platzhalter für zufällige ID Auswahl
            audio, sr = sdl.loadRawWithID(ID)

            # zerstückeln des Audios:
            CHUNK = 16384
            prediction2 = []
            for a_chunk in [audio[i*CHUNK:(i+1)*CHUNK] for i in range(len(audio)//CHUNK)]: 
                features = sdl.extractFeaturesFromAudio(np.array(a_chunk), sr = sr, drop_beg=0, drop_end=2)

                for f in features.values:
                    prediction = clf.predict(f.reshape(1,-1))
                    prediction2.append(prediction)
                    from collections import Counter
                    if len(prediction>1):
                        print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction]))
                    else:
                        print("\nPREDICTION:\t", Counter(class_names[int(prediction)]))
                    time.sleep(0.5)
            print("\nPREDICTION:\t", Counter([class_names[int(p)] for p in prediction2]))
            print("Platzhalter für Asphalt!")

        ### GUI
        top = tk.Tk()
        # Code to add widgets will go here...
        button_Asphalt = tk.Button(top, text="Asphalt", fg="red")
        button_Asphalt.bind("<Button-1>", startAsphalt)


        button_Asphalt.pack()
        top.mainloop()