import math
import re
from stemming.porter2 import stem

def tf(document):
    '''Get the TF array from document based on the terms in document

    Use double normalization 0.5

    Args:
        document (array): array of words separated by paragraphs

    Return:
        dict: word maps to frequency
    '''
    result = {}
    _max = 0
    for i in document:
        result[i] = result.get(i, 0) + 1
    for i in result.values():
        _max = _max if i <= _max else i
    for i in result:
        result[i] = 0.5 + 0.5*result[i]/_max
    return result

def idf(term, tfs):
    '''calculate the IDF value

    Args:
        term (string): the term to query
        tfs (array): TF results of documents for words in each document

    Returns:
        float: IDF value
    '''
    count = 0
    N = len(tfs)
    for i in tfs:
        if i.get(term, 0.0) > 0.0:
            count += 1
    return math.log(1.0*(N+1)/(1+count))

def tfidf(index, documents):
    '''This is an implementation for using TF-IDF algorithm.

    Actually tfidf here is used to create scores for a set of terms.
    LSI would be a better method together with TF-IDF, but require
    extra memory and Matrix multiplication/transportation/eigen-value
    computation. We use some dummy weighted addition to get the final
    score.

    Args:
        index (int): the index in documents for querying
        documents (array): a set of words separated by each paragraph

    Returns:
        array: Contains (index, score) pair for all the other documents
    '''
    tfs = []
    query = documents.pop(index)
    _mp = tf(query)
    query = _mp.keys()
    for i in documents:
        tfs.append(tf(i))
    idfs = map(lambda x:idf(x, tfs) ,query)
    scores = []
    for k in xrange(len(documents)):
        score = 0
        for i, j in enumerate(idfs):
            score += _mp[query[i]] * j * tfs[k].get(query[i], 0)
        if k >= index:
            scores.append((k+1, score))
        else:
            scores.append((k, score))
    return scores

def paragraph_to_words(para):
    '''Process paragraph to words

    Remove separate words, and process with stemming algorithm

    Args:
        para (string): paragraph as string

    Returns:
        array: words array
    '''
    return filter(lambda x: len(x) > 0,
            map(lambda x: stem(x), re.split('[ ,.]+', para)))

if __name__ == '__main__':
    documents = map(lambda x:
      paragraph_to_words(x),
    [
        "I have a pen, I have a book",
        "My name is Pencil",
        "I have books, book",
        "I have booked."
    ])
    print(documents)
    print(tfidf(0, documents))
