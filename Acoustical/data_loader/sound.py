from base.base_data_loader import BaseDataLoader
import warnings
import random
import os
import pandas as pd
import numpy as np
import librosa
from utils.config import process_config
from utils.dirs import listdir
import utils.utils as utils
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
import time

## testing
from multiprocessing import Pool, cpu_count
##


class SoundDataLoader(BaseDataLoader):
    def __init__(self, config="configs/urbanSound.json"):
        if isinstance(config, str):
            config = process_config(config)
        super(SoundDataLoader, self).__init__(config)

        # get the absolute path of the datastore
        currentDrive, path = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-1]),"Datastore","Acoustical")

        self.attributes = pd.read_csv(os.path.join(dataFolder, self.config.testplan), sep=";")
        if not os.path.isfile(os.path.join(dataFolder, self.config.features)):
            print("features.csv not found...")
            print("\nLoading features...")
            self.createFeatureFrame(dataFolder)

        self.features = pd.read_csv(os.path.join(dataFolder, self.config.features), sep=";")
        self.data = None
        pd.options.mode.chained_assignment = None  # default='warn'
            
    def loadFeature_csv(self, path):
        """ Load the extracted Features from a .csv-File to the class attribute self.features. """
        dataType = ".csv"
        if not path.endswith(dataType):
            print("File %s is NOT a %s file!" % path,dataType)
            exit(0)
        path = os.path.normpath(path)
        pathSplit = path.split(os.sep)
        fileID = pathSplit[-1].split("_")[0]
        df_temp = pd.read_csv(path,";")
        if not "frame" in df_temp.columns:
            df_temp = df_temp.rename(columns={"Unnamed: 0":"frame"})
        else:
            df_temp = df_temp.drop(columns=["Unnamed: 0"])
        self.features = utils.pdStackWithNone(self.features,df_temp,fileID)

    def getFeaturesWithLabel(self, attribute, label):
        """ Returns the Pandas-Frame with the features corresponding to the given attributes and labels. 
            
        Parameters
        ----------
        attribute : string or list, same length as ''label'' if list
        label     : string or list, same length as ''attribute'' if list
        """


        if not isinstance(attribute,list):
            attribute = [attribute]
        if not isinstance(label,list):
            label = [label]

        ID_list = None
        temp_list = None
        temp_list2 = None
        for a,l in zip(attribute,label):
            if not isinstance(l, list):
                l = [l]
            if l[0] is None:
                l = self.attributes[a].unique()
            ID_list = None
            for ll in l:
                if ID_list is None:
                    ID_list = self.attributes.ID[self.attributes[a] == ll].values
                else:
                    ID_list = np.concatenate((ID_list,self.attributes.ID[self.attributes[a] == ll].values))
            if temp_list is None:
                temp_list = ID_list
            else:
                temp_list = list(set(temp_list) & set(ID_list))

        return self.features[self.features.ID.isin(temp_list)]
    
    def equalize(self, frame, attribute="Belag", randomize=False, split_train_test=None):
        # Vorteil von split_train_test in sdl equalize: jede ID wird abhängig von dem equalizen gesplittet.
        # Bsp.: ID 170 (Beton, nass) besteht aus 55 frames. Insgesamt gibt es von der Klasse (Beton, nass) 2784 frames.
        # Am wenigsten frames gibt es von der Klasse (Asphalt, trocken): 1874 frames.
        # Nach dem equalizen sind es also noch 55 * 1874 / 2784 = 36 frames (Zufällig ausgewählte, wenn randomize True. Sonst die ersten).
        # Bei einem split_train_test=0.7 sind es nach dem split noch 36*7/10=25 frames im train und 11 frames im test DataFrame von der ID 170.
        # Hierdurch wird verhindert, dass der Zufall beim .sample() einige IDs komplett entfernt.
        attributes = attribute.copy()
        if isinstance(attributes,list) and len(attributes) > 1: # Sind es mehrere Attribute? (bzw. eine Liste)
            att1 = attributes[0]
            att2 = attributes[1]
            att = [att1,att2]
            combinedAttributes = ",".join(att)
            if len(attributes) > 2:
                frame2 = self.equalize(frame, att,randomize=randomize, split_train_test=None)
                return self.equalize(frame2, [combinedAttributes] + attributes[2:], randomize=randomize, split_train_test=split_train_test) ###

            # alle Attribute in die Featureliste einfügen
            for A in att:
                if A not in frame:
                    frame[A] = self.getAttributeToFeatures(frame,A)

            # Bestimmen der attributes-Kombination mit den wenigsten Frames
            
            frame[combinedAttributes] = self.combineAttributes(frame,attribute)
            frame = frame.drop(columns=att)
            
            class_counts = frame[combinedAttributes].value_counts()
            min_classCounts = min(class_counts)
            min_className = class_counts.index[np.where(class_counts == min_classCounts)]
            df_temp = pd.DataFrame()
            
            for cA in frame[combinedAttributes].unique():
                actualFrames = frame[frame[combinedAttributes]==cA]                        
                # equalize IDs
                quot = min_classCounts/actualFrames.shape[0]
                if quot <= 0.1:
                    print("\n","#" * 50)
                    print("Warning: The class(",cA,") has more than 10x the amount of frames availiable than the class with the lowest frameset (", min_className, actualFrames.shape[0],"/",min_classCounts,").")
                    print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
                    print("Consider setting other Filter settings to reduce the amount of frames of the class(",cA,").")
                    print("#" * 50,"\n")
                IDs = actualFrames.ID.unique()
                for id in IDs:
                    if randomize:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(round(quot * actualFrames[actualFrames.ID == id].shape[0])))], 
                                            ignore_index=True)
                    else:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                            ignore_index=True)

        else: # ...ist nur ein attribute
            df_temp = pd.DataFrame()            
            if isinstance(attributes, list):
                attributes = attributes[0]
            frame[attributes] = self.getAttributeToFeatures(frame, attributes)
            class_names = frame[attributes].unique()
            min_classCounts = min(frame[attributes].value_counts())
            for cn in class_names:
                actualFrames = frame[frame[attributes]==cn]
                        
                # equalize IDs
                quot = min_classCounts/actualFrames.shape[0]
                if quot <= 0.1:
                    print("\n","#" * 100)
                    print("Warning: The class(",cn,") has more than 10x the amount of frames availiable than the class with the lowest frameset (", min_className,actualFrames.shape[0],"/",min_classCounts,").")
                    print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
                    print("Consider setting other Filter settings to reduce the amount of frames of the class(",cn,").")
                    print("#" * 100,"\n")

                IDs = actualFrames.ID.unique()
                for id in IDs:
                    if randomize:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(quot * actualFrames[actualFrames.ID == id].shape[0]))], 
                                            ignore_index=True)
                    else:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                            ignore_index=True)

        if split_train_test is not None:
            if split_train_test <= 0 or split_train_test > 1:
                raise ValueError("split_train_test has to be a value (float) between 0 and 1")
            
            from sklearn.model_selection import train_test_split
            train = pd.DataFrame()
            test = pd.DataFrame()
            IDs = df_temp.ID.unique()
            for id in IDs:
                tr, te = train_test_split(df_temp[df_temp.ID == id], test_size=(1-split_train_test) )
                train = pd.concat((train, tr), ignore_index = True)
                test = pd.concat((test, te), ignore_index = True)
                
            #return train.reset_index(), test.reset_index()
            return train, test

        else:
            return df_temp
    
    def bad_equalize(self, frame, attribute="Belag", randomize=False, split_train_test=None):
        # Vorteil von split_train_test in sdl equalize: jede ID wird abhängig von dem equalizen gesplittet.
        # Bsp.: ID 170 (Beton, nass) besteht aus 55 frames. Insgesamt gibt es von der Klasse (Beton, nass) 2784 frames.
        # Am wenigsten frames gibt es von der Klasse (Asphalt, trocken): 1874 frames.
        # Nach dem equalizen sind es also noch 55 * 1874 / 2784 = 36 frames (Zufällig ausgewählte, wenn randomize True. Sonst die ersten).
        # Bei einem split_train_test=0.7 sind es nach dem split noch 36*7/10=25 frames im train und 11 frames im test DataFrame von der ID 170.
        # Hierdurch wird verhindert, dass der Zufall beim .sample() einige IDs komplett entfernt.
        attributes = attribute.copy()
        if isinstance(attributes,list) and len(attributes) > 1: # Sind es mehrere Attribute? (bzw. eine Liste)
            att1 = attributes[0]
            att2 = attributes[1]
            att = [att1,att2]
            combinedAttributes = ",".join(att)
            if len(attributes) > 2:
                frame2 = self.equalize(frame, att,randomize=randomize, split_train_test=None)
                return self.equalize(frame2, [combinedAttributes] + attributes[2:], randomize=randomize, split_train_test=split_train_test) ###

            # alle Attribute in die Featureliste einfügen
            for A in att:
                if A not in frame:
                    frame[A] = self.getAttributeToFeatures(frame,A)

            # Bestimmen der attributes-Kombination mit den wenigsten Frames
            
            frame[combinedAttributes] = self.combineAttributes(frame,attribute)
            frame = frame.drop(columns=att)
            
            class_counts = frame[combinedAttributes].value_counts()
            min_classCounts = min(class_counts)
            df_temp = pd.DataFrame()
            
            pcs = min(len(frame[combinedAttributes].unique()), cpu_count()-1)
            pool = Pool(pcs)
            cAs = [frame[combinedAttributes].unique()[i::pcs] for i in range(pcs)]
            ti = time.time()
            t = pool.map(self.multi, [[frame, min_classCounts, combinedAttributes, randomize, cA] for cA in cAs])
            print(time.time() - ti)
            df_temp = pd.concat(t, ignore_index=True)

            #for cA in frame[combinedAttributes].unique():
            #    actualFrames = frame[frame[combinedAttributes]==cA]                        
            #    # equalize IDs
            #    quot = min_classCounts/actualFrames.shape[0]
            #    if quot <= 0.1:
            #        print("\n","#" * 100)
            #        print("Warning: The class(",cA,") has more than 10x the amount of frames availiable than the class with the lowest frameset (",actualFrames.shape[0],"/",min_classCounts,").")
            #        print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
            #        print("Consider setting other Filter settings to reduce the amount of frames of the class(",cA,").")
            #        print("#" * 100,"\n")
            #    IDs = actualFrames.ID.unique()
            #    for id in IDs:
            #        if randomize:
            #            df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(round(quot * actualFrames[actualFrames.ID == id].shape[0])))], 
            #                                ignore_index=True)
            #        else:
            #            df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
            #                                ignore_index=True)

        else: # ...ist nur ein attribute
            df_temp = pd.DataFrame()            
            if isinstance(attributes, list):
                attributes = attributes[0]
            frame[attributes] = self.getAttributeToFeatures(frame, attributes)
            class_names = frame[attributes].unique()
            min_classCounts = min(frame[attributes].value_counts())
            for cn in class_names:
                actualFrames = frame[frame[attributes]==cn]
                        
                # equalize IDs
                quot = min_classCounts/actualFrames.shape[0]
                if quot <= 0.1:
                    print("\n","#" * 100)
                    print("Warning: The class(",cn,") has more than 10x the amount of frames availiable than the class with the lowest frameset (",actualFrames.shape[0],"/",min_classCounts,").")
                    print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
                    print("Consider setting other Filter settings to reduce the amount of frames of the class(",cn,").")
                    print("#" * 100,"\n")

                IDs = actualFrames.ID.unique()
                for id in IDs:
                    if randomize:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(quot * actualFrames[actualFrames.ID == id].shape[0]))], 
                                            ignore_index=True)
                    else:
                        df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                            ignore_index=True)

        if split_train_test is not None:
            if split_train_test <= 0 or split_train_test > 1:
                raise ValueError("split_train_test has to be a value (float) between 0 and 1")
            
            from sklearn.model_selection import train_test_split
            train = pd.DataFrame()
            test = pd.DataFrame()
            IDs = df_temp.ID.unique()
            for id in IDs:
                tr, te = train_test_split(df_temp[df_temp.ID == id], test_size=(1-split_train_test) )
                train = pd.concat((train, tr), ignore_index = True)
                test = pd.concat((test, te), ignore_index = True)
                
            #return train.reset_index(), test.reset_index()
            return train, test

        else:
            return df_temp.reset_index()#.drop(columns = attribute),_
    
    def multi(self, input):
        frame, min_classCounts, combinedAttributes, randomize, cA = input
        #print("\n\n\n\n\n############################\n############################\n############################\n############################\n############################\n\n\n\n")
        actualFrames = frame[frame[combinedAttributes]==cA[0]]                        
        # equalize IDs
        quot = min_classCounts/actualFrames.shape[0]
        if quot <= 0.1:
            print("\n","#" * 100)
            print("Warning: The class(",cA,") has more than 10x the amount of frames availiable than the class with the lowest frameset (",actualFrames.shape[0],"/",min_classCounts,").")
            print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
            print("Consider setting other Filter settings to reduce the amount of frames of the class(",cA,").")
            print("#" * 100,"\n")
        IDs = actualFrames.ID.unique()
        df_temp = []
        for id in IDs:
            if randomize:
                df_temp.append(actualFrames[actualFrames.ID == id].sample(int(round(quot * actualFrames[actualFrames.ID == id].shape[0]))))
            else:
                df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                    ignore_index=True)

        return df_temp
        
    def old_multi(self, input):
        frame, min_classCounts, combinedAttributes, randomize, cA = input
        print("\n\n\n\n\n############################\n############################\n############################\n############################\n############################\n\n\n\n")
        actualFrames = frame[frame[combinedAttributes]==cA[0]]                        
        # equalize IDs
        quot = min_classCounts/actualFrames.shape[0]
        if quot <= 0.1:
            print("\n","#" * 100)
            print("Warning: The class(",cA,") has more than 10x the amount of frames availiable than the class with the lowest frameset (",actualFrames.shape[0],"/",min_classCounts,").")
            print("This might lead to an unbalanced representation of the IDs in the training/test seperation.")
            print("Consider setting other Filter settings to reduce the amount of frames of the class(",cA,").")
            print("#" * 100,"\n")
        IDs = actualFrames.ID.unique()
        df_temp = pd.DataFrame()
        for id in IDs:
            if randomize:
                df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(round(quot * actualFrames[actualFrames.ID == id].shape[0])))], 
                                    ignore_index=True)
            else:
                df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                    ignore_index=True)

        return df_temp

    def concatFrames(self, input):
        #input = input[0] # Für Solocore
        IDs_chunk, actualFrames, quot, min_classCounts, randomize = input
        # Funktion um das concatten mehrkernfähig zu machen
        df_temp = pd.DataFrame()
        t = time.time()
        for id in IDs_chunk:
            if randomize:
                df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].sample(int(round(quot * actualFrames[actualFrames.ID == id].shape[0])))], 
                                    ignore_index=True)
            else:
                df_temp = pd.concat([df_temp, actualFrames[actualFrames.ID == id].head(min_classCounts)], 
                                    ignore_index=True)
        print(time.time()-t,"\ncopy")
        return df_temp

    def getAttributeToFeatures(self, frame, attribute):
        return frame.ID.apply(lambda x: self.attributes[attribute][x])

    def changeLabel_ofIDs(self, IDs, attribute, label):
        self.attributes[attribute][self.attributes.ID.isin(IDs)] = label

    def extractFeaturesFromAudio(self, audio, n_fft = 16384, sr=48000):
        #stft = librosa.stft(audio,n_fft=n_fft)
        #freqs = np.abs(stft)
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=None)

        chroma_stft      = librosa.feature.chroma_stft(audio, sr = sr, n_fft=16384, hop_length=int(n_fft/4))
        chroma_cqt      = librosa.feature.chroma_cqt(audio, sr = sr, hop_length=int(n_fft/4)) 
        chroma_cens      = librosa.feature.chroma_cens(audio, sr = sr, hop_length=int(n_fft/4)) 
        mel         = librosa.feature.melspectrogram(audio,sr, n_fft=n_fft, hop_length=int(n_fft/4))
        mfcc         = librosa.feature.mfcc(audio,sr) #
        rmse        = librosa.feature.rmse(audio, frame_length=n_fft, hop_length=int(n_fft/4))
        centroid    = librosa.feature.spectral_centroid(audio,sr, n_fft=n_fft, hop_length=int(n_fft/4))
        bandwidth   = librosa.feature.spectral_bandwidth(audio,sr, n_fft=n_fft, hop_length=int(n_fft/4))
        contrast    = librosa.feature.spectral_contrast(audio,sr, n_fft=n_fft, hop_length=int(n_fft/4))
        flatness    = librosa.feature.spectral_flatness(audio, n_fft=n_fft, hop_length=int(n_fft/4))
        rolloff     = librosa.feature.spectral_rolloff(audio,sr,n_fft=n_fft,hop_length=int(n_fft/4))
        poly_features     = librosa.feature.poly_features(audio,sr,n_fft=n_fft,hop_length=int(n_fft/4)) 
        tonnetz     = librosa.feature.tonnetz(audio, sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft,hop_length=int(n_fft/4)) 
        return pd.DataFrame(np.concatenate((chroma_stft.T,chroma_cqt.T,chroma_cens.T,mel.T,mfcc.T[:chroma_stft.shape[1]],rmse.T,centroid.T,bandwidth.T,
                                            contrast.T,flatness.T,rolloff.T,
                                            #poly_features.T,
                                            tonnetz.T[:chroma_stft.shape[1]], zero_crossing_rate.T),axis=1)
                                           ,None,
                                           ["chroma_stft_"+str(i) for i in range(chroma_stft.shape[0])]+
                                           ["chroma_cqt_"+str(i) for i in range(chroma_cqt.shape[0])]+
                                           ["chroma_cens_"+str(i) for i in range(chroma_cens.shape[0])]+
                                           ["mel_"+str(i) for i in range(mel.shape[0])]+
                                           ["mfcc_"+str(i) for i in range(mfcc.shape[0])]+
                                           ["rmse_"+str(i) for i in range(rmse.shape[0])]+
                                           ["centroid_"+str(i) for i in range(centroid.shape[0])]+
                                           ["bandwidth_"+str(i) for i in range(bandwidth.shape[0])]+
                                           ["contrast_"+str(i) for i in range(contrast.shape[0])]+
                                           ["flatness_"+str(i) for i in range(flatness.shape[0])]+
                                           ["rolloff_"+str(i) for i in range(rolloff.shape[0])]+
                                           #["poly_features_"+str(i) for i in range(poly_features.shape[0])]+
                                           ["tonnetz_"+str(i) for i in range(tonnetz.shape[0])]+
                                           ["zero_crossing_rate_"+str(i) for i in range(zero_crossing_rate.shape[0])])

    def combineAttributes(self, frame, attributes, sep = ","):
        return frame[attributes].astype(str).apply(lambda x: sep.join(x), axis=1)

    def createFeatureFrame(self, path):
        # Erstellen des Features-Dataframes.
        # Nutzt (Anzahl der CPU-Kerne)-1 Kerne.
        # Speichert das DataFrame unter dem in self.config.features hinterlegten Pfad.
        pcs = cpu_count()-1
        print("Using",pcs,"Cores....")
        actualDataFolder = "1_set"
        files = listdir(os.path.join(path,actualDataFolder))
        chunks = [files[i::pcs] for i in range(pcs)]

        pool = Pool(processes=pcs)

        result = pool.map(self.combineExtractedFeaturesFromAudio, [[os.path.join(path,actualDataFolder), chunk] for chunk in chunks])
        #result = self.combineExtractedFeaturesFromAudio([[os.path.join(path,actualDataFolder), chunk] for chunk in chunks])

        features = pd.concat(result, ignore_index=True)
        features.to_csv(os.path.join(path, self.config.features), sep=";", index=False)

    def combineExtractedFeaturesFromAudio(self, input):
        # Zusammenführen von mehreren Feature-Frames.
        # Funktion wurde implementiert um das Zusammenzuführen mehrkernfähig zu machen (Siehe self.createFeatureFrame)
        path = input[0]
        files = input[1]
        features = pd.DataFrame()
        for f in files:
            print(f)
            newFeatures = self.extractFeaturesFromAudio(os.path.join(path,f))
            newFeatures.insert(0, "frame", range(len(newFeatures)))
            newFeatures.insert(0, "ID", f.split("_")[0])
            features = pd.concat((features, newFeatures), ignore_index=True)#, copy=False)
        return features

    def loadRawWithID(self,ID):
        currentDrive, path = os.path.splitdrive(os.getcwd())
        dataFolder = os.path.join(currentDrive,os.path.sep.join(path.split(os.path.sep)[:-1]),"Datastore","Acoustical")
        path = os.path.join(dataFolder, "1_set")
        files = listdir(path)
        files_ID_dict = dict(zip([int(ID) for ID in [f.split("_")[0] for f in files]],files))
        return sf.read(os.path.join(path,files_ID_dict[ID]))

    def loadRawWithPath(self, path):
        return sf.read(path)

    def Attr_to_class(self, frame, label="Belag"):
        if isinstance(label, list) and len(label) > 1:
            subclass1 = frame[label[0]].unique()
            subclass2 = frame[label[1]].unique()
            class_names = []
            for s1 in subclass1:
                for s2 in subclass2:
                    class_names.append([s1,s2])
            classes_list = []
            for i,row in frame.iterrows():
                classes_list.append(class_names.index(list(row[label].values)))
            return classes_list, class_names
        else:
            if isinstance(label, list):
                label = label[0]
            class_names = list(frame[label].unique())
            return frame[label].apply(lambda x: class_names.index(x)), class_names
    
    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, verbose=0):

        import itertools
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            if verbose>10: print("Normalized confusion matrix")
        else:
            if verbose>10: print('Confusion matrix, without normalization')

        if verbose>10: print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')