# -*- coding: utf-8 -*-
# Define A NaiveBayes class which is used to distinguish between languages using a character bigram model.

# Do create necessary variables wherever required. You might have to add self in some definitions and calls and keep variables in self as well

from collections import defaultdict, Counter
from operator import itemgetter
from math import log
from typing import DefaultDict
import sys
import io
# Define A NaiveBayes class which is used to distinguish between languages using a character bigram model.

# Do create necessary variables wherever required.

import enum
import re

class NaiveBayes():
    def extract_ngrams(self,x: str, n=2) -> "list[str]":
        ###TODO###
        #extract character ngrams
        words = x.split()
        n_grams = []
        pattern_punctuation = re.compile(r'[^\w\s]') # only character + digit + space


        for i in range(len(words) - n + 1):
            n_gram = " ".join(words[i:i+n])
            input_string = n_gram.lower()
            output_string = pattern_punctuation.sub('', input_string)

            n_grams.append(output_string)

        return n_grams
        
    def smoothed_log_likelihood(self, w: str, c: str, k: int, count: 'DefaultDict[str, Counter]', vocab: "set[str]", denominator) -> float:
        ###TODO###
        #apply smoothing

        # numerator
        if count[w] == []:
          numerator = 1
        else:
          numerator = count[w][c] + 1

        # denominator
        return log(numerator/denominator)
        

    def train_nb(self, docs: "list[tuple[str, str]]", k: int = 1, n: int = 2) -> "tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]":
        ###TODO###
        """
        Train a Naive-Bayes model

        :param docs: The documents, each associated with a clas label (document, label)
        :type docs: list[tuple[str, str]]
        :param k: The value added to the numerator and denominator to smooth likelihoods
        :type k: int
        :para n: the order of ngrams.
        :type b: int
        :return: The log priors, log likelihoods, the classes, and the vocabulary for the model at a tuple
        :rtype: tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]
        """
        ## Initialize vocab (the vocabulary) and docs_by_class (a dictionary in which
        ## class labels are keys and lists of documents are values).

        vocab = set()
        classes = set()
        docs_by_class = DefaultDict(list)

        ## Populate vocab and docs_by_class.
        for idx, (c, doc) in enumerate(docs): #class and doc            
            ## Represent documents as collections of ngrams (default n=2)
            ngrams = self.extract_ngrams(doc)
            vocab |= set(ngrams)
            docs_by_class[c].append(ngrams)
            classes.add(c)

            

          
        ## Total number of documents in the collection
        num_docs = idx + 1 #len(docs)
        #print("num_docs",num_docs)

        ## Number of documents in each class (with each label)
        num_docs_in_class = {} #{key:len(value) for key,value in docs_by_class.items()}
        #print("num_docs_in_class", num_docs_in_class)

        ## The log priors for each class
        # dict[str, float]
        log_prior = {} #{key: value/num_docs for key, value in num_docs_in_class.items()}
        
        ## Counts of times a label c occurs with an ngram w
        # 'DefaultDict[str, Counter]'
        count = DefaultDict(list)
        inverse_count = DefaultDict(list) 

        # DefaultDict[str, DefaultDict[str, float]],
        log_likelihood = DefaultDict(list)

        for c, documents in docs_by_class.items():
            # Iterate over the documents of class c, accumulating <ngram, class> counts in counts
            inverse_count[c] = Counter() 
            for d in documents:
                for w in d:
                  if len(count[w]) == 0:
                    count[w] = Counter()
                  count[w][c] += 1
                  inverse_count[c][w] += 1

            # Calculate the number of documents with class label c
            num_docs_in_class[c]= len(documents)

            ## Calculate the log prior as the log of the number of documents in class c over the total number of documents
            #dict[str, float]
            log_prior[c] = log(num_docs_in_class[c] / num_docs)

            ## Calculate the log likelihood for each (w|c). Smooth by k=1.
            denominator =  sum(inverse_count[c].values()) + len(inverse_count[c].keys())

            for w in vocab:
              if len(log_likelihood[w]) == 0:
                log_likelihood[w] = DefaultDict(float)
              log_likelihood[w][c] = self.smoothed_log_likelihood(w, c, k, count, vocab, denominator)
        
        return log_prior, log_likelihood, classes, vocab

    def classify(self, testdoc: str, log_prior: "dict[str, float]", log_likelihood: "DefaultDict[str, DefaultDict[str, float]]", classes: "set[str]", vocab: "set[str]", k: int=1, n: int=2) -> str:
        ###TODO###
        """Given a trained NB model (log_prior, log_likelihood, classes, and vocab), returns the most likely label for the input document.

        :param textdoc str: The test document.
        :param log_prior dict[str, float]: The log priors of each category. Categories are keys and log priors are values.
        :param log_likelihood DefaultDict[str, DefaultDict[str, float]]: The log likelihoods for each combination of word/ngram and class.
        :param classes set[str]: The set of class labels (as strings).
        :param vocab set[str]: The set of words/negrams in the vocabulary.
        :param k int: the value added in smoothing.
        "param n int: the order of ngrams.
        :return: The best label for `testdoc` in light of the model.
        :rtype: str
        """
        ## Extract a set of ngrams from `testdoc`
        #self.classify(doc, log_prior, log_likelihood, classes, vocab, n = n)
        doc = self.extract_ngrams(testdoc)
        #print("doc",doc)

        ## Initialize the sums for each class. These will be the "scores" based on which class will be assigned.

        class_sum = DefaultDict(float)
        #print("log_prior",log_prior)
        #print("log_likelihood",log_likelihood)
        
        ## Iterate over the classes, computing `class_sum` for each
        for c in classes:
            ## Initialize `class_sum` with the log prior for the class
            class_sum[c] = log_prior[c]

            ## Then add the likelihood for each in-vocabulary ngram in the document
            for i, w in enumerate(doc):
                if w in vocab:
                    try:
                        class_sum[c] += log_likelihood[w][c]
                        
                    except ValueError:
                      pass
                
                        
                        # count = 
                        # count[w] = 
                        # log_likelihood[w][c] = 
                        # class_sum[c] = 

        return max(class_sum.items(), key=itemgetter(1))[0]
    
    def precision(self,tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        return tp / (tp + fp)

    def recall(self,tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        return tp / (tp + fn)

    def micro_precision(self, tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        tp_sum = sum(tp.values())
        fp_sum = sum(fp.values())
        return tp_sum / (tp_sum + fp_sum)

    def micro_recall(self, tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        tp_sum = sum(tp.values())
        fn_sum = sum(fn.values())
        return tp_sum / (tp_sum + fn_sum)

    def micro_f1(self, tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
        mp = self.micro_precision(tp, fp)
        mr = self.micro_recall(tp, fn)
        return 2 * (mp * mr) / (mp + mr)

    def macro_precision(self, tp: "dict[str, int]", fp: "dict[str, int]") -> float:
        n = len(tp)
        return (1 / n) * sum([self.precision(tp[c], fp[c]) for c in tp.keys()])

    def macro_recall(self, tp: "dict[str, int]", fn: "dict[str, int]") -> float:
        n = len(tp)
        return (1 / n) * sum([self.recall(tp[c], fn[c]) for c in tp.keys()])

    def macro_f1(self, tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
        n = len(tp)
        return 2 * (self.macro_precision(tp, fp) * self.macro_recall(tp, fn)) / (self.macro_precision(tp, fp) + self.macro_recall(tp, fn))

    def evaluate(self, train: "list[tuple[str, str]]", eval: "list[tuple[str, str]]", n: int=2):
        log_prior, log_likelihood, classes, vocab = self.train_nb(train, n = n)
        # Initialize dictionaries for true positives, false positives, and false negatives
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))
        for c_ref, doc in eval:
            c_hyp = self.classify(doc, log_prior, log_likelihood, classes, vocab, n = n)
            confusion[c_ref][c_hyp] += 1
            if c_ref == c_hyp:
                tp[c_ref] += 1
            else:
                fn[c_ref] += 1
                fp[c_hyp] += 1

        return self.macro_precision(tp, fp), self.macro_recall(tp, fn), self.macro_f1(tp, fp, fn), self.micro_precision(tp, fp), self.micro_recall(tp, fn), self.micro_f1(tp, fp, fn)


"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with open(train_txt) as f:
        train = [tuple(l.split('\t')) for l in f]
    
    with open(test_txt) as f:
        test = [tuple(l.split('\t')) for l in f]

    tmp=NaiveBayes()
    map, mar, maf, mp, mr, mf=tmp.evaluate(train, test, n=2)
   

