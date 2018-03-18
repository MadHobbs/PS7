"""
Author      : Yi-Chieh Wu
Class       : HMC, CS 158
Date        : 2015 Aug 06
Description : ML utilities
"""

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

######################################################################
# global settings
######################################################################

mpl.lines.width = 2
mpl.axes.labelsize = 14


######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename, converters=None) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename   -- string, filename
            converters -- dict, optional
                          see numpy.loadtxt(...)
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",", converters=converters)
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self) :
        """Plot features and labels."""
        pos = np.nonzero(self.y > 0)  # matlab: find(y > 0)
        neg = np.nonzero(self.y < 0)  # matlab: find(y < 0)
        plt.plot(self.X[pos,0], self.X[pos,1], 'b+', markersize=5)
        plt.plot(self.X[neg,0], self.X[neg,1], 'ro', markersize=5)
        plt.show()

# helper functions
def load_data(filename, converters=None) :
    """Load csv file into Data class."""
    data = Data()
    data.load(filename, converters)
    return data