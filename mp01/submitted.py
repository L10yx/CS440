'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter



def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # raise RuntimeError("You need to write this part!")
    
    word_counts = [text.count(word0) for text in texts]
    cX0 = max(word_counts, default = 0) + 1
    Pmarginal = np.zeros(cX0)
    for count in word_counts:
        Pmarginal[count] += 1
    Pmarginal /= len(texts)
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    word_counts0 = [text.count(word0) for text in texts]
    word_counts1 = [text.count(word1) for text in texts]
    cX0 = max(word_counts0, default = 0) + 1
    cX1 = max(word_counts1, default = 0) + 1
    Pcond = np.zeros((cX0, cX1))
    for (x0, x1) in zip(word_counts0, word_counts1):
        Pcond[x0, x1] += 1
    for x0 in range(cX0):
        sum_by_x0 = np.sum(Pcond[x0, :])
        Pcond[x0, :] = Pcond[x0, :]/sum_by_x0 if sum_by_x0 > 0 else np.nan

    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    Pjoint = Pcond
    for x0 in range(len(Pmarginal)):
        Pjoint[x0, :] = Pjoint[x0, :] * Pmarginal[x0] if Pmarginal[x0] else 0
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    # raise RuntimeError("You need to write this part!")
    cX0, cX1 = Pjoint.shape
    mu = [0, 0]
    for x0 in range(cX0):
        for x1 in range(cX1):
            mu += Pjoint[x0, x1] * np.array([x0, x1])
        
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    # raise RuntimeError("You need to write this part!")
    cX0, cX1 = Pjoint.shape
    Sigma = np.zeros((2,2))
    for x0 in range(cX0):
        for x1 in range(cX1):
            Sigma[0, 0] += Pjoint[x0, x1] * (x0 - mu[0]) * (x0 - mu[0])
            Sigma[1, 1] += Pjoint[x0, x1] * (x1 - mu[1]) * (x1 - mu[1])
            Sigma[0, 1] += Pjoint[x0, x1] * (x0 - mu[0]) * (x1 - mu[1])
    Sigma[1, 0] = Sigma[0, 1]
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    # raise RuntimeError("You need to write this part!")
    cX0, cX1 = Pjoint.shape
    Pfunc = Counter()

    for x0 in range(cX0):
        for x1 in range(cX1):
            z = f(x0, x1)
            Pfunc[z] += Pjoint[x0, x1]
    return Pfunc
    
