'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")
    word_tag_counts = defaultdict(Counter)
    tag_overall_counts = defaultdict(int)
    
    # Count word-tag occurrences in training data
    for sentence in train:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
            tag_overall_counts[tag] += 1

    # Find the most frequent tag overall
    most_frequent_tag = max(tag_overall_counts, key=tag_overall_counts.get)

    # Find the most frequent tag for each word
    most_frequent_tags = {}
    for word, tag_counts in word_tag_counts.items():
        most_frequent_tags[word] = tag_counts.most_common(1)[0][0]

    # Tag the test data using the most frequent tag for each word
    tagged_test = []
    for sentence in test:
        tagged_sentence = [(word, most_frequent_tags.get(word, most_frequent_tag)) for word in sentence]
        tagged_test.append(tagged_sentence)

    return tagged_test

# def viterbi(test, train):
#     '''
#     Implementation for the viterbi tagger.
#     input:  test data (list of sentences, no tags on the words)
#             training data (list of sentences, with tags on the words)
#     output: list of sentences with tags on the words
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''
#     # raise NotImplementedError("You need to write this part!")
#     # Step 1: Count occurrences of tags, tag pairs, tag/word pairs
#     # Initialize count dictionaries
#     tag_count = defaultdict(int)
#     tag_pair_count = defaultdict(int)
#     tag_word_count = defaultdict(lambda: defaultdict(int))
    
#     # Count occurrences in the training data
#     for sentence in train:
#         prev_tag = 'START'
#         for word, tag in sentence:
#             tag_count[tag] += 1
#             tag_pair_count[(prev_tag, tag)] += 1
#             tag_word_count[tag][word] += 1
#             prev_tag = tag
    
#     # Step 2: Compute smoothed probabilities
#     # Initialize probability dictionaries
#     num_tags = len(tag_count)
#     num_words = sum(len(sen) for sen in train)
#     #print(num_tags)
#     alpha = 0.000001  # Laplace smoothing constant
    
#     # Initial probabilities (transition from START)
#     initial_prob = {tag: (tag_pair_count[('START', tag)] + alpha) / (sum([tag_pair_count[('START', tag)] for tag in tag_count]) + alpha * (num_tags + 1)) for tag in tag_count}
#     #print(sum(len(sen) for sen in train))
#     #print(initial_prob)
#     #print(sum(initial_prob.values()))
#     # Transition probabilities
#     transition_prob = {(prev_tag, tag): (tag_pair_count[(prev_tag, tag)] + alpha) / (num_words + alpha * (num_tags + 1)) for (prev_tag, tag) in tag_pair_count}

#     #print(sum(transition_prob.values()))
#     # Emission probabilities
#     emission_prob = {tag: {word: (tag_word_count[tag][word] + alpha) / (num_words + alpha * (num_tags + 1)) for word in tag_word_count[tag]} for tag in tag_count}
    
#     # Add a default emission probability for unknown words
#     for tag in tag_count:
#         emission_prob[tag]['UKNNOWN'] = alpha / (num_words + alpha * (num_tags + 1))
#     #print(sum(sum(word.values()) for word in emission_prob.values()))
#     # Step 3: Take the log of each probability
#     initial_prob_log = {tag: log(prob) for tag, prob in initial_prob.items()}
#     transition_prob_log = {(prev_tag, tag): log(prob) for (prev_tag, tag), prob in transition_prob.items()}
#     emission_prob_log = {tag: {word: log(prob) for word, prob in word_probs.items()} for tag, word_probs in emission_prob.items()}
#     #print(sum(sum(word.values()) for word in emission_prob_log.values()))
#     # Step 4: Construct the trellis
#     tagged_sentences = []
#     for sentence in test:
#         trellis = [{}]
#         backpointer = [{}]
        
#         # Initialize the trellis for the start of the sentence
#         for tag in tag_count:
#             trellis[0][tag] = initial_prob_log[tag] + emission_prob_log[tag].get(sentence[0], emission_prob_log[tag]['UKNNOWN'])
#             backpointer[0][tag] = 'START'
        
#         # Forward pass
#         for t in range(1, len(sentence)):
#             trellis.append({})
#             new_backpointer = {}
#             for tag in tag_count:
#                 max_prob = float('-inf')
#                 best_prev_tag = None
#                 for prev_tag in trellis[t - 1]:
#                     prob = trellis[t - 1][prev_tag] + transition_prob_log.get((prev_tag, tag), float('-inf')) + emission_prob_log[tag].get(sentence[t], emission_prob_log[tag]['UKNNOWN'])
#                     if prob > max_prob:
#                         max_prob = prob
#                         best_prev_tag = prev_tag
#                 trellis[t][tag] = max_prob
#                 new_backpointer[tag] = best_prev_tag
#             backpointer.append(new_backpointer)
        
#         # Backward pass to find the best path
#         best_path = []
#         max_prob = float('-inf')
#         best_tag = None
#         for tag in trellis[-1]:
#             if trellis[-1][tag] > max_prob:
#                 max_prob = trellis[-1][tag]
#                 best_tag = tag
#         best_path.append(best_tag)
#         prev_tag = best_tag
#         for t in range(len(sentence) - 1, 0, -1):
#             prev_tag = backpointer[t][prev_tag]
#             best_path.insert(0, prev_tag)
        
#         # Tag the sentence with the best path
#         tagged_sentence = [(word, best_path[i]) for i, word in enumerate(sentence)]
#         tagged_sentences.append(tagged_sentence)
#         #print(tagged_sentence)
    
#     return tagged_sentences
def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Count occurrences of tags, tag pairs, tag/word pairs
    tag_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    for sentence in train:
        prev_tag = 'START'
        for word, tag in sentence:
            tag_counts[tag] += 1
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            prev_tag = tag
    

    alpha = 0.001
    num_tags = len(tag_counts)
    num_words = sum(len(sentence) for sentence in train)
    
    initial_probs = {tag: (tag_counts[tag] + alpha) / (num_words + alpha * num_tags) for tag in tag_counts}
    transition_probs = {(prev_tag, tag): (count + alpha) / (tag_counts[prev_tag] + alpha * num_tags)
                       for (prev_tag, tag), count in transition_counts.items()}
    emission_probs = {(tag, word): (count + alpha) / (tag_counts[tag] + alpha * num_words)
                     for (tag, word), count in emission_counts.items()}
    
    # Take the log of each probability
    initial_probs_log = {tag: math.log(prob) for tag, prob in initial_probs.items()}
    transition_probs_log = {(prev_tag, tag): math.log(prob) for (prev_tag, tag), prob in transition_probs.items()}
    emission_probs_log = {(tag, word): math.log(prob) for (tag, word), prob in emission_probs.items()}
    
    # Construct the trellis
    tagged_sentences = []
    for sentence in test:
        trellis = [{}]
        for tag in tag_counts:
            if (tag, sentence[0]) in emission_probs_log:
                trellis[0][tag] = {
                    'prob': initial_probs_log[tag] + emission_probs_log[(tag, sentence[0])],
                    'prev': None
                }
            else:
                trellis[0][tag] = {
                    'prob': initial_probs_log[tag] + math.log(alpha / (tag_counts[tag] + alpha * num_words)),
                    'prev': None
                }
        
        for t in range(1, len(sentence)):
            trellis.append({})
            for tag in tag_counts:
                max_prob = float('-inf')
                prev_tag_selected = None
                for prev_tag in tag_counts:
                    if (prev_tag, tag) in transition_probs_log and (tag, sentence[t]) in emission_probs_log:
                        prob = trellis[t - 1][prev_tag]['prob'] + transition_probs_log[(prev_tag, tag)] + \
                               emission_probs_log[(tag, sentence[t])]
                    else:
                        prob = trellis[t - 1][prev_tag]['prob'] + transition_probs_log.get((prev_tag, tag), math.log(alpha / (tag_counts[prev_tag] + alpha * num_tags))) + \
                               emission_probs_log.get((tag, sentence[t]), math.log(alpha / (tag_counts[tag] + alpha * num_words)))
                    
                    if prob > max_prob:
                        max_prob = prob
                        prev_tag_selected = prev_tag
                
                trellis[t][tag] = {'prob': max_prob, 'prev': prev_tag_selected}
        
        #  Return the best path through the trellis
        best_path = []
        max_prob = float('-inf')
        best_tag = None
        for tag in tag_counts:
            if trellis[-1][tag]['prob'] > max_prob:
                max_prob = trellis[-1][tag]['prob']
                best_tag = tag
        
        for t in range(len(sentence) - 1, -1, -1):
            best_path.insert(0, (sentence[t], best_tag))
            best_tag = trellis[t][best_tag]['prev']
        
        tagged_sentences.append(best_path)
    
    return tagged_sentences
def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    hapax_counts = defaultdict(int)

    # Count word occurrences and identify hapax words
    for sentence in train:
        for word, tag in sentence:
            tag_counts[tag] += 1
            emission_counts[(tag, word)] += 1
            if word not in hapax_counts:
                hapax_counts[word] = 0
            hapax_counts[word] += 1

    hapax_tag_probs = defaultdict(float)
    for word, count in hapax_counts.items():
        for tag in tag_counts:
            if (tag, word) in emission_counts:
                hapax_tag_probs[tag] += 1

    # Normalize hapax tag probabilities
    for tag in tag_counts:
        hapax_tag_probs[tag] /= sum(hapax_tag_probs.values())

    # Laplace smoothing constant
    alpha = 0.001

    # Compute smoothed emission probabilities
    num_tags = len(tag_counts)
    num_words = sum(tag_counts.values())

    emission_probs = {}
    for (tag, word), count in emission_counts.items():
        hapax_prob = hapax_tag_probs[tag]
        emission_probs[(tag, word)] = (count + alpha) / (tag_counts[tag] + alpha * num_words) * hapax_prob

    # Construct trellis
    tagged_sentences = []
    for sentence in test:
        trellis = [{}]

        # Initialize trellis for the first word
        for tag in tag_counts:
            if (tag, sentence[0]) in emission_probs:
                trellis[0][tag] = {
                    'prob': math.log(emission_probs[(tag, sentence[0])]),
                    'prev': None
                }
            else:
                trellis[0][tag] = {
                    'prob': math.log(alpha / (tag_counts[tag] + alpha * num_words)),
                    'prev': None
                }

        # Fill in the trellis for subsequent words
        for t in range(1, len(sentence)):
            trellis.append({})
            for tag in tag_counts:
                max_prob = float('-inf')
                prev_tag_selected = None
                for prev_tag in tag_counts:
                    if (prev_tag, tag) in transition_counts and (tag, sentence[t]) in emission_probs:
                        prob = trellis[t - 1][prev_tag]['prob'] + math.log(transition_counts[(prev_tag, tag)]) + math.log(emission_probs[(tag, sentence[t])])
                    else:
                        prob = trellis[t - 1][prev_tag]['prob'] + math.log(alpha / (tag_counts[prev_tag] + alpha * num_tags))

                    if prob > max_prob:
                        max_prob = prob
                        prev_tag_selected = prev_tag

                trellis[t][tag] = {'prob': max_prob, 'prev': prev_tag_selected}

        # Find the best path through the trellis
        best_path = []
        max_prob = float('-inf')
        best_tag = None
        for tag in tag_counts:
            if trellis[-1][tag]['prob'] > max_prob:
                max_prob = trellis[-1][tag]['prob']
                best_tag = tag

        for t in range(len(sentence) - 1, -1, -1):
            best_path.insert(0, (sentence[t], best_tag))
            best_tag = trellis[t][best_tag]['prev']

        tagged_sentences.append(best_path)

    return tagged_sentences