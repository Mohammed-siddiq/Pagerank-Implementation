# Overview

Developed as part of my master's course CS 582 - Information retrieval - at University of Illinois,Chicago

This project implements pagerank algorithm for the given collection of documents. 
Page rank is run over individual documents and its corresponding MRR is calculated for top k phrases (k ranging from 1 to 10), using the corresponding ground truth file.


# Dependencies and steps to run

The following libraries are used :

- pandas
- nltk
- numpy

Use the following command to run

`python pagerankImplementation.py`

requires an input for the path containing the abstract(all documents POS tagged ) and gold directories:

for example pass `/Users/mohammedsiddiq/Downloads/www` if it contains the abstract and gold directories.

# Implementation

- Takes the input path containing the directories abstract and gold.
- Takes the window size as the input path
- Reads all the files from the abstract and gold directories.
- Preprocess individual documents form the abstract directory.
  
  The following pre-processing is done:
  
  - First the document is tokenized
  - POS tags are extracted
  - stop words are removed
  - stemming is performed
  
- Gold files are also prep-processed, only stemming is performed

- For each document the following steps are applied to calculate the rank:
   
    - Number of words occuring together in the window of each word is found
    - This provides the weight for the word in the document
    - A graph representation is constructed with words as nodes with their weights
    - The above graph is represented as a matrix
    - Initial probability is set for the the above graph representation.
    - damping factor is considered to be 0.85
    - Final matrix is constructed to find the page rank
    - rank probabilities are constructed for fixed number of iterations ( assumed to be 10)
    - Finally for each of these documents ranked words, ngrams are constructed(uni/bi/tri).
    - Score of the ngrams are updated as the sum of ranks of individual words
    - Ngrams are ranked
    - These ranked ngrams are used to calculate the MRR at top k phrases (k=> 1 to 10)
    
- The final mrr is calculated by averaging on the collection size.

# Results

This implementation gave the following MRR for the given collection of documents

 MRR @ k= 1  :  0.0
 
 MRR @ k= 2  :  0.004815650865312265
 
 MRR @ k= 3  :  0.008954100827689992
 
 MRR @ k= 4 :   0.011959592359378506
 
 MRR @ k= 5  :  0.016732246341964765
 
 MRR @ k= 6  :  0.021999364475900045
 
 MRR @ k= 7  :  0.0268212857199432
 
 MRR @ k= 8  :  0.030941820314518427
 
 MRR @ k= 9  :  0.037431092266544355
 
 MRR @ k= 10 :  0.0445237863795545


