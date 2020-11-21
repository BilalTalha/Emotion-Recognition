import os
import pickle
import numpy as np


class SimpleDatsetLoader:

    def __init__(self,dataPaths):
        self.dataPaths = dataPaths

    # ================================= DEAP dataset =============================================================== #
    def load_DEAP(self):

        DEAP = {}

        data = []
        labels = []

        for i, file in enumerate(os.listdir(self.dataPaths)):
            filename = os.path.join(self.dataPaths, file)
            raw_data = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
            X = raw_data["data"]
            y = raw_data["labels"]

            data.append(X)
            labels.append(y)

        data   = np.array(data)
        labels = np.array(labels)
        # data   = np.reshape(data,   (32,40, data.shape[2],data.shape[3]))
        # labels = np.reshape(labels, (32,40, labels.shape[2]))
        channels_name = np.array(["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5",
                                  "CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4",
                                  "Fz","F4","F8","Fc6","Fc2","Cz","C4","T8", "CP6","CP2","P4","P8",
                                  "PO4","O2","hEOG","vEOG","zEMG","tEMG","GSR",
                                  "Respiration belt","Plethysmograph","Temprature"])
        
        labels_name = np.array(["valence", "arousal", "dominance", "liking"])

        DEAP = {"data":data, "labels":labels, "labels_name":labels_name, "channels_name":channels_name}

        return(DEAP)

    # ============================== DEAP dataset With Two Classes=========================================== #

    def load_DEAP_BinaryClass(self, valence_threshold=5, arousal_threshold=5):
        DEAP = self.load_DEAP()

        data   = DEAP["data"]
        labels = DEAP["labels"]

        no_sample = data.shape[0]
        no_trial  = data.shape[1]
        no_chan   = data.shape[2]
        no_pnts   = data.shape[3]
        no_label  = labels.shape[2]

        labels_name = np.array(["valence", "arousal"])

        y = np.zeros((no_sample*no_trial,len(labels_name)))
        y_temp = labels.reshape((no_sample*no_trial,no_label))
        for i in range(no_sample*no_trial):

            # Valence Labels
            if y_temp[i,0] < valence_threshold:
                y[i,0] = 0
            else:
                y[i,0] = 1

            # Arousal labels
            if y_temp[i,1] < arousal_threshold:
                y[i,1] = 0
            else:
                y[i,1] = 1
        y = np.reshape(y,(no_sample,no_trial,2))

        # Information About the data=============
        print("""
         [INFO]--->
         This dataset has been constructed as [subject, trials, channels, data] and labels are [subject, trial,            valence, arousal].
         label = 0 ---> low  | valence rating < {}   | arousal rating < {} 
         label = 1 ---> high | valence rating > {}   | arousal rating > {}
         
         The shape of each item inside the dataset is as following:
         Data.shape     : {}
         labels.shape   : {}
         first column of labels belongs to  : {}
         second column of labels belongs to : {}
         """.format(valence_threshold,arousal_threshold,valence_threshold,arousal_threshold,data.shape,y.shape,
                    labels_name[0],labels_name[1]))

        result = {"data": data, "labels": y, "labels_name": labels_name, "channels_name": DEAP["channels_name"]}

        return(result)

    # ============================== DEAP dataset With Three Classes========================================= #

    def load_DEAP_TripleClass(self,valence_threshold_low=4, valence_threshold_high=6, arousal_threshold_low =4,           arousal_threshold_high=6):
        DEAP = self.load_DEAP()

        data = DEAP["data"]
        labels = DEAP["labels"]

        no_sample = data.shape[0]
        no_trial = data.shape[1]
        no_chan = data.shape[2]
        no_pnts = data.shape[3]
        no_label = labels.shape[2]

        labels_name = np.array(["valence", "arousal"])

        y = np.zeros((no_sample * no_trial, len(labels_name)))
        y_temp = labels.reshape((no_sample * no_trial, no_label))
        for i in range(no_sample * no_trial):

            # Valence Labels
            # low valence
            if y_temp[i, 0] < valence_threshold_low:
                y[i, 0] = 0
            # medium valence
            elif valence_threshold_low <= y_temp[i, 0] <= valence_threshold_high:
                y[i, 0] = 1
            # high valence
            else:
                y[i,0] = 2

            # Arousal labels
            # low arousal
            if y_temp[i, 1] < arousal_threshold_low:
                y[i, 1] = 0
            # medium arousal
            elif arousal_threshold_low <= y_temp[i,1] <= arousal_threshold_high:
                y[i,1] = 1
            # high arousal
            else:
                y[i, 1] = 2
        y = np.reshape(y, (no_sample, no_trial, 2))
        # Information About the data=============
        print("""
         [INFO]--->
         This dataset has been constructed as [subject, trials, channels, data] and labels are [subject, trial,            valence, arousal].
         label = 0 ---> low    | valence: Negative  | arousal: Passive 
         label = 1 ---> medium | valence: Neutral   | arousal: Neutral  
         label = 2 ---> high   | valence: Positive  | arousal: Active

         The shape of each item inside the dataset is as following:
         Data.shape     : {}
         labels.shape   : {}
         first column of labels belongs to  : {}
         second column of labels belongs to : {}
         """.format(data.shape,y.shape,labels_name[0],labels_name[1]))


        result = {"data": data, "labels": y, "labels_name": labels_name, "channels_name": DEAP["channels_name"]}

        return (result)


    # ============================== DEAP dataset With Four Classes======================================= #
    def load_DEAP_FourthClass(self, arousal_threshold=5, valence_threshold=5):
        DEAP = self.load_DEAP()

        data = DEAP["data"]
        labels = DEAP["labels"]

        no_sample = data.shape[0]
        no_trial  = data.shape[1]
        no_chan   = data.shape[2]
        no_pnts   = data.shape[3]
        no_label  = labels.shape[2]
        
        labels_name = np.array(["HAHV", "HALV", "LALV","LAHV"])

        y = np.zeros((no_sample * no_trial,))
        y_temp = labels.reshape((no_sample * no_trial, no_label))
        for i in range(no_sample * no_trial):

            # HAHV
            if (y_temp[i, 0] >= valence_threshold) and (y_temp[i, 1] >= arousal_threshold):
                y[i] = 0
            # HALV
            elif (y_temp[i, 0] < valence_threshold) and (y_temp[i, 1] >= arousal_threshold):
                y[i] = 1
            # LALV
            elif(y_temp[i, 0] < valence_threshold) and (y_temp[i, 1] < arousal_threshold):
                y[i] = 2
            else:
                y[i] = 3

        y = np.reshape(y, (no_sample, no_trial))
        # Information About the data=============
        print("""
         [INFO]--->
         This dataset has been constructed as [subject, trials, channels, data] and labels are [subject, trial].
         label = 0 ---> HAHV    | valence rating >= {}    | arousal rating >= {} 
         label = 1 ---> HALV    | valence rating <  {}    | arousal rating >= {} 
         label = 2 ---> LALV    | valence rating <  {}    | arousal rating <  {}
         label = 3 ---> LAHV    | valence rating >= {}    | arousal rating <  {}

         The shape of each item inside the dataset is as following:
         Data.shape     : {}
         labels.shape   : {}
         """.format(valence_threshold,arousal_threshold,valence_threshold,
                    arousal_threshold,valence_threshold,arousal_threshold,
                    valence_threshold,arousal_threshold,
                    data.shape,y.shape))


        result = {"data": data, "labels": y, "labels_name": labels_name, "channels_name": DEAP["channels_name"]}

        return (result)