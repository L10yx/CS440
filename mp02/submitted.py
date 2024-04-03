'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter
import math
from tqdm import tqdm

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    # raise RuntimeError("You need to write this part!")
    frequency = {}
    for y in train:
        frequency[y] = Counter()
        for text in train[y]:
            bigrams = ['*-*-*-*'.join(text[i:i+2]) for i in range(len(text)-1)]
            frequency[y] += Counter(bigrams)

    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    # raise RuntimeError("You need to write this part!")
    nonstop = {}
    for y in frequency:
        nonstop[y] = Counter()
        for bigram, count in frequency[y].items():
            word1, word2 = bigram.split('*-*-*-*')
            if word1 not in stopwords or word2 not in stopwords:
                nonstop[y][bigram] = count

    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    # raise RuntimeError("You need to write this part!")
    likelihood = {}
#     vocabulary = set()
#     for y in nonstop:
#         vocabulary.update(nonstop[y])
    
    for y in nonstop:
        likelihood[y] = {}
        total_bigrams = sum(nonstop[y].values())
        total_types = len(nonstop[y])
        for bigram in nonstop[y]:
            likelihood[y][bigram] = (nonstop[y][bigram] + smoothness) / (total_bigrams + smoothness * (total_types + 1))
        likelihood[y]['OOV'] = smoothness / (total_bigrams + smoothness * (total_types + 1))

    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    # raise RuntimeError("You need to write this part!")
    hypotheses = []
    for text in texts:
        max_posterior = float('-inf')
        best_class = None
        pos_posterior = float('-inf')
        neg_posterior = float('-inf')
        for class_label, class_likelihood in likelihood.items():
            log_posterior = math.log(prior) if class_label == 'pos' else math.log(1 - prior)
            for i in range(len(text) - 1):
                
                if text[i] in stopwords and text[i + 1] in stopwords:
                    continue
                bigram = text[i] + '*-*-*-*' + text[i + 1]
                if bigram in class_likelihood:
                    log_posterior += math.log(class_likelihood[bigram])
                else:
                    log_posterior += math.log(class_likelihood['OOV'])
            if class_label == 'pos':
                pos_posterior = log_posterior
            else:
                neg_posterior = log_posterior
        if pos_posterior > neg_posterior:
            best_class = 'pos'
        elif pos_posterior == neg_posterior:
            best_class = 'undecided'
        else:
            best_class = 'neg'
        hypotheses.append(best_class)

    return hypotheses


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    # raise RuntimeError("You need to write this part!")


    accuracies = np.zeros((len(priors), len(smoothnesses)))

    for i, prior in enumerate(priors):
        for j, smoothness in enumerate(smoothnesses):
            predictions = []
            pos_likelihood = math.log(prior)
            neg_likelihood = math.log(1 - prior)
            pos_denominator = sum(nonstop['pos'].values()) + smoothness * (len(nonstop['pos']) + 1)
            neg_denominator = sum(nonstop['neg'].values()) + smoothness * (len(nonstop['neg']) + 1)
            for text in texts:
                for k in range(len(text) - 1):
                    bigram = text[k] + '*-*-*-*' + text[k + 1]
                    if text[k] in stopwords and text[k + 1] in stopwords:
                        continue
                    if bigram in nonstop['pos']:
                        pos_likelihood += math.log((nonstop['pos'][bigram] + smoothness)) - math.log(pos_denominator)
                    else:
                        pos_likelihood += math.log(smoothness) - math.log(pos_denominator)

                    if bigram in nonstop['neg']:
                        neg_likelihood += math.log((nonstop['neg'][bigram] + smoothness)) - math.log(neg_denominator)
                    else:
                        neg_likelihood += math.log(smoothness) - math.log(neg_denominator)

                prediction = 'pos' if pos_likelihood > neg_likelihood else 'neg'
                if pos_likelihood == neg_likelihood:
                    prediction = 'undecided'
                predictions.append(prediction)
                pos_likelihood = math.log(prior)
                neg_likelihood = math.log(1 - prior)
               
            correct = sum(p == l for p, l in zip(predictions, labels))
            accuracies[i, j] = correct / len(texts)

    return accuracies
    
                          