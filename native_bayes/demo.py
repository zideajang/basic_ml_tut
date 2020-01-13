# coding=utf-8
from nltk.corpus import movie_reviews
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def main():
    print(movie_reviews.categories())

if __name__ == '__main__':
    main()
    