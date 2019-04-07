
# coding: utf-8

# In[1]:



'''
Name: Mohammed Siddiq
UID:  msiddi56@uic.edu
UIN:  664750555
'''
import sys

import pandas as pd
import os
import nltk
import re
import itertools
import operator
from nltk.stem.porter import *
from nltk.corpus import stopwords
import math
import numpy as np
import string
import collections


# In[2]:


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
file_path = input("Enter the file path containing the documents:\t")
expected_pos = ["NN", "NNS", "NNP", "NNPS", "JJ"]
custom_window_size = int(input("Enter the window size:\t"))
counter = 0
failed_counter = 0


# In[3]:


def load_documents(path):
    df = pd.DataFrame()
    abstract_path = path + "//abstracts"
    print("loading abstract documents ...")
    print()
    abstract_docs = os.listdir(abstract_path)
    read_counter = 0
    for doc in abstract_docs:
        with open(abstract_path + "//" + doc, 'r') as content_file:
            content = content_file.read()
        #             print(content)
        df = df.append({'abstract': content, 'docId': doc}, ignore_index=True)
        read_counter += 1
        # print(read_counter // len(abstract_docs) * 100, "% done..", flush=True)

    # loading corresponding gold file
    gold_path = path + "//gold"
    gold_docs = os.listdir(gold_path)
    print("loading gold documents ...")
    print()
    read_counter = 0
    for doc in gold_docs:
        with open(gold_path + "//" + doc, 'r') as content_file:
            content = content_file.read()
            df.loc[df.docId == doc, 'gold'] = content
        read_counter += 1
        # print(read_counter // len(gold_docs) * 100, "% done..", flush=True)
    return df


# df.shape


# df.count()
# df.loc[df['docId'] == '9466892']


def remove_punctions(word):
    for c in word:
        if c in string.punctuation:
            word = word.replace(c, '')
    return word.strip()


def preprocess(document):
    #     print("Preprocessing abstract documents..")
    words = document.split()
    processed_words = []
    for word in words:
        words_with_pos = word.split("_")  # extract the Pos after _
        #         words_with_pos = [remove_punctions(word) for word in words_with_pos]
        if len(words_with_pos) == 2:
            if len(words_with_pos[0]) > 1 and words_with_pos[0] not in stop_words and words_with_pos[1] in expected_pos:
                final_word = ps.stem(words_with_pos[0])
                processed_words.append(final_word)
    return processed_words


# In[4]:


def preprocess_gold(document):
    #     print("Preprocessing(Stemming) gold documents...")
    sentences = document.split("\n")
    preprocessed_doc = []
    for sentence in sentences:
        words = sentence.split()
        stemmed_sentence = ''
        for word in words:
            if len(word.strip()) > 0 and word not in stop_words:
                final_word = ps.stem(word)
                if len(stemmed_sentence) == 0:  # first word of the sentence
                    stemmed_sentence += final_word
                else:
                    stemmed_sentence += ' ' + final_word
        preprocessed_doc.append(stemmed_sentence)
    return preprocessed_doc


# In[5]:


def build_wighted_graph(document_words, window_size):
    graph_rep = collections.defaultdict(lambda: collections.Counter())
    for i, current_word in enumerate(document_words):
        for windowI in range(i - window_size, i + window_size + 1):  # check in the window of the word
            if windowI >= 0 and windowI < len(document_words):  # boundary checks for the window
                if windowI != i:  # avoiding self match with the word
                    window_word = document_words[windowI]
                    graph_rep[current_word][
                        window_word] += 1  # updating the counter if the current word co occurs with another word in the window
    return graph_rep


# In[6]:


def extract_bag_of_words(graph_rep_df):
    bow = set()
    bow.update(list(graph_rep_df.columns.values))  # adding all columns
    bow.update(list(graph_rep_df.index.values))  # adding all the rows
    return bow


# In[7]:


def fill_missing_series(df, bow):
    for word in bow:
        if not word in df:
            df[word] = pd.Series(0.0, index=df.index)
    return df


# In[8]:


def construct_matrix(df, bow):
    df = df.copy()
    df = fill_missing_series(df, bow)  # fill missing columns
    df = fill_missing_series(df.T, bow).T  # fill missing rows
    return df.fillna(0.0)  # replace Nan's with 0


# In[9]:


def create_links_for_unliked(doc_matrix):
    doc_matrix = doc_matrix.T  # transposing to get the columns
    for column in doc_matrix:
        if doc_matrix[column].sum() == 0.0:
            doc_matrix[column] = pd.Series(np.ones(len(doc_matrix[column])), index=doc_matrix.index)
    return doc_matrix.T 


# In[10]:


def set_initial_probablity(word_nodes):
    #     print("words " , word_nodes)
    initial_probability_val = 1.0 / float(len(word_nodes))
    # creating a series representing initial probabilities for each word in the document
    initial_probabilites = pd.Series({node: initial_probability_val for node in word_nodes})
    return initial_probabilites


# In[11]:


def normalize_values_as_probability(doc_matrix):
    return doc_matrix.div(doc_matrix.sum(axis=1), axis=0)


# In[12]:


def update_final_probability(word_nodes, alpha, initial_probabilities):
    random_jump_probability = 1.0 / len(word_nodes) * (1 - alpha)
    final_probability = initial_probabilities.copy().multiply(alpha) + random_jump_probability
    return final_probability


# In[13]:


def form_ngrams_and_score(document, ranked_words):
    ranked_phrases = {}
    # updating unigrams score
    for word in document:
        ranked_phrases[word] = ranked_words.get(word)
    bigrams = list(nltk.ngrams(document, 2))
    trigrams = list(nltk.ngrams(document, 3))

    # updating bigrams score
    for w1, w2 in bigrams:
        ranked_phrases[w1 + " " + w2] = ranked_words.get(w1) + ranked_words.get(w2)
    # updating trigrams score
    for w1, w2, w3 in trigrams:
        ranked_phrases[w1 + " " + w2 + " " + w3] = ranked_words.get(w1) + ranked_words.get(w2) + ranked_words.get(w3)
    phrases_with_scroes = sorted(ranked_phrases.items(), key=operator.itemgetter(1), reverse=True)
    phrases = [phrase for phrase, rank in phrases_with_scroes]
    rank = [rank for phrase, rank in phrases_with_scroes]
    return phrases, rank


# In[14]:


counter = 0
failed_counter = 0


# In[15]:


def apply_page_rank(document_words, alpha, number_of_iterations, window_size, docId):
    # build_wighted_graph
    # print("building weighted graph for document : " + docId)
    global counter
    global failed_counter

    graph_ds = build_wighted_graph(document_words=document_words, window_size=window_size)
    if len(graph_ds) != 0:
        df_matrix = pd.DataFrame(graph_ds)

        document_bow = extract_bag_of_words(df_matrix)

        df_matrix = construct_matrix(df_matrix, document_bow)
        df_matrix = create_links_for_unliked(df_matrix)

        # print("Initialiazing matrix:")
        # print(df_matrix)

        rank = set_initial_probablity(document_bow)
        df_matrix_final = normalize_values_as_probability(df_matrix)
        df_matrix_final = update_final_probability(word_nodes=document_bow, alpha=alpha,
                                                   initial_probabilities=df_matrix_final)
        # print("Calculating final rank")
        # print(df_matrix_final)
        for i in range(number_of_iterations):
            rank = rank.dot(df_matrix_final)

        ranked_phrases, ranked_scores = form_ngrams_and_score(document=document_words, ranked_words=rank)
        counter += 1
        return ranked_phrases
    else:
        failed_counter += 1
        return []


# In[16]:


df = load_documents(file_path)
print("Done loading documents...")
df = df.dropna()
print(df.head())


# In[17]:


print("Preprocessing abstract documents...")
df['abstract'] = df.abstract.apply(preprocess)


# In[18]:


print(df.head())
print("preprocessing [Stemming] gold documents....")
df['gold'] = df.gold.apply(preprocess_gold)


# In[19]:


MRR_collection = [0.0 for i in range(10)]


# In[20]:


print("Applying page rank for individual documents...")
df['ranked_phrases'] = df.abstract.apply(apply_page_rank, alpha=0.85, number_of_iterations=10, window_size=custom_window_size,
                                         docId=df['docId'])


# In[21]:


print("ranked phrases...")
print(df.ranked_phrases)


# In[22]:


def update_MRR_document(ranked_phrases, MRR_collection, gold_phrases):
    for top in range(10):
        MRR_at_k = 0.0
        for k in range(top):  # finding hits in the top k
            if k < len(ranked_phrases) and ranked_phrases[k] in gold_phrases:
                rank = gold_phrases.index(ranked_phrases[k]) + 1
                MRR_at_k += 1.0 / rank
                break;
        MRR_collection[top] += MRR_at_k
    return MRR_collection


# In[23]:


print("calculating MRR")
MRR_collection = [0.0 for i in range(10)]
for ranked_phrases,gold in zip(df['ranked_phrases'],df['gold']):
    MRR_collection = update_MRR_document(MRR_collection=MRR_collection,gold_phrases=gold,ranked_phrases=ranked_phrases)


# In[24]:


MRR_collection = [1/counter * mrr_at_k for mrr_at_k in MRR_collection]


# In[25]:


for k in range(10):
    print(" MRR @ k=", k + 1, ": ", MRR_collection[k])

