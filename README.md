# Spam-Ham-Classifier 
## A Machine learning classifier to predict whether the SMS is Spam or Ham by using Natural Language Processing (NLP)

## Table of Content

  * [Dataset](#dataset)
  * [Demo](#demo)
  * [Screenshots](#screenshots)
  * [Methodology](#methodology)
  * [Bug / Feature Request](#bug--feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  
## Dataset
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

Dataset link : https://www.kaggle.com/uciml/sms-spam-collection-dataset

## Demo
Link: [https://spam-ham-nlp-model.herokuapp.com/](https://spam-ham-nlp-model.herokuapp.com/)

## Screenshots

#### CHECKING PART 1 

<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(69).png"></a>
<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(70).png"></a>

#### CHECKING PART 2

<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(72).png"></a>
<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(71).png"></a>

## Methodology

### 1. Create Virtual Environment
It is always a good practise to create a virtual environment and mostly it is very useful while deploying app. To create virtual environment for python and jupyter follow this link : https://janakiev.com/blog/jupyter-virtual-envs/

### 2. (Natural Language Toolkit) NLTK:
NLTK is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks.
Installing NLTK Library
```bash
!pip install nltk 
```
Type above code in the Jupyter Notebook or if it doesn’t work, type this in your cmd prompt "pip install nltk". This should work in most cases. Install NLTK: http://pypi.python.org/pypi/nltk

Importing NLTK Library

-> import nltk

-> nltk.download()

Download the packages.

### 3. Reading and Exploring Dataset

#### Reading in text data & why do we need to clean the text?
While reading data, we get data in the structured or unstructured format. A structured format has a well-defined pattern whereas unstructured data has no proper structure. In between the 2 structures, we have a semi-structured format which is a comparably better structured than unstructured format. When we read semi-structured data it is hard to interpret so we use pandas to easily understand our data.

### 4. Pre-processing Data
Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:

##### i. Remove punctuation
Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you

##### ii.Tokenization
Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.

##### iii. Remove stopwords
Stopwords are common words that will likely appear in any text. They don’t tell us much about our data so we remove them. eg: silver or lead is fine for me-> silver, lead, fine.

#### 5. Preprocessing Data: Stemming
Stemming helps reduce a word to its stem form. It often makes sense to treat related words in the same way. It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl Note: Some search engines treat words with the same stem as synonyms.

#### 6. Preprocessing Data: Lemmatizing
Lemmatizing derives the canonical form (‘lemma’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle In Short, Stemming is typically faster as it simply chops off the end of the word, without understanding the context of the word. Lemmatizing is slower and more accurate as it takes an informed analysis with the context of the word in mind.

#### 7. Vectorizing Data
Vectorizing is the process of encoding text as integers i.e. numeric form to create feature vectors so that machine learning algorithms can understand our data.

##### i. Vectorizing Data: Bag-Of-Words
Bag of Words (BoW) or CountVectorizer describes the presence of words within the text data. It gives a result of 1 if present in the sentence and 0 if not present. It, therefore, creates a bag of words with a document-matrix count in each text document.

##### ii. Vectorizing Data: N-Grams
N-grams are simply all combinations of adjacent words or letters of length n that we can find in our source text. Ngrams with n=1 are called unigrams. Similarly, bigrams (n=2), trigrams (n=3) and so on can also be used.

Unigrams usually don’t contain much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the letter or word is likely to follow the given word. The longer the n-gram (higher n), the more context you have to work with.

##### iii. Vectorizing Data: TF-IDF
It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents). Note: Used for search engine scoring, text summarization, document clustering.

TF-IDF is applied on the body_text, so the relative count of each word in the sentences is stored in the document matrix. (Check the repo). Note: Vectorizers outputs sparse matrices. Sparse Matrix is a matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the locations of the non-zero elements.
