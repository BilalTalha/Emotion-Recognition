import pandas as pd
import numpy as np 
import pymrmr

class ChannelSelection:
    
    """ This class must be called after loading the dataset
        and defining X which was a 4 dimensional array in the DEAP dataset. The valence
        must also be obtained from the dataset which was a 2D array reshaped in the function
        to a 1D array since together with data of 32 channels and valence as the target variable
        a dataframe will be created on which mRMR is implemented.
        
    """
    def __init__(self, x, target, channel, scheme):#X is the channel data,target is 2D array for valence,and channel is no. of best channels we need
        
        self.x = x
        self.target = target
        self.channel = channel
        self.scheme = scheme
    
    def mrmr(self): 
        
        if self.x.shape[2] == 32:
            mean_x = np.mean(self.x,axis=3,keepdims=True) 
            mean_x = mean_x.reshape(mean_x.shape[0]*mean_x.shape[1],mean_x.shape[2])
            cols = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5",
                    "CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4",
                    "Fz","F4","F8","Fc6","Fc2","Cz","C4","T8", "CP6","CP2","P4","P8",
                    "PO4","O2"]                 #names of 32 channels

            df = pd.DataFrame(mean_x, columns=cols)

            self.target = self.target.reshape(self.target.shape[0]*self.target.shape[1],1)

            df["Class"] = self.target
            df["Class"] = df["Class"].astype(int)
            col = df.columns.tolist()
            col = col[-1:] + col[:-1] #arranging the sequence of columns so class (target) is in first column and the rest are features (channels)
            df = df[col]

            return pymrmr.mRMR(df,self.scheme,self.channel)#mRMR can be "MID" or "MIQ"
        
        elif self.x.shape[2] == 62:
            mean_x = np.mean(self.x,axis=3,keepdims=True) 
            mean_x = mean_x.reshape(mean_x.shape[0]*mean_x.shape[1],mean_x.shape[2])
            cols = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1",
             "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1",
             "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
             "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
             "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3",
             "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
             "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]                 #names of 62 channels

            df = pd.DataFrame(mean_x, columns=cols)

            self.target = self.target.reshape(self.target.shape[0]*self.target.shape[1],1)

            df["Class"] = self.target
            df["Class"] = df["Class"].astype(int)
            col = df.columns.tolist()
            col = col[-1:] + col[:-1] #arranging the sequence of columns so class (target) is in first column and the rest are features (channels)
            df = df[col]

            return pymrmr.mRMR(df,self.scheme,self.channel)#mRMR can be "MID" or "MIQ"
            

class Channels:
    
    def __init__(self, mrmr_output):#X is the channel data,target is 2D array for valence,and channel is no. of best channels we need
        
        self.mrmr_output = mrmr_output
        
        
    def channel_index(self):
        
        cols = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5",
                "CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4",
                "Fz","F4","F8","Fc6","Fc2","Cz","C4","T8", "CP6","CP2","P4","P8",
                "PO4","O2"]
        channels = {} #dictionary to store index of channels
        for i in range(len(cols)):
            channels[i] = cols[i]
        
        ch_index = []
        keys = list(channels.keys())
        values = list(channels.values())
        for i in self.mrmr_output:
            ch_index.append(keys[values.index(i)])
        
        return ch_index

