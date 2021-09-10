import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from pymorphy2 import MorphAnalyzer
import re


class Lemmatizer(object):
    def __init__(self, detect_stop_words=False, detect_punctuation=False):
        self.detect_stop_words = detect_stop_words
        if (detect_stop_words):
            self.stopwords_list = stopwords.words("russian")
        self.detect_punctuation = detect_punctuation

    def lemmatize(self, word):
        pass


class StupidLemmatizer(Lemmatizer):
    def __init__(self, detect_stop_words=False, detect_punctuation=True):
        super().__init__(detect_stop_words, detect_punctuation)
        self.mystem = Mystem()

    def lemmatize(self, text):
        words = self.mystem.lemmatize(text)
        words = [word for word in words if word != '\n' and word != ' ']
        if (self.detect_punctuation):
            words = [word for word in words if word.strip() not in punctuation]
        if (self.detect_stop_words):
            words = [word for word in words if word not in self.stopwords_list]
        return words


class PyMorphyLemmatizer(Lemmatizer):
    def __init__(self, detect_stop_words=False, detect_punctuation=True):
        super().__init__(detect_stop_words, detect_punctuation)
        self.pymorphy_analyzer = MorphAnalyzer()

    def lemmatize(self, text):
        words = re.findall(r"[\w']+|[!\"#$%&'()*+,\-./:;<=>?@[/\]^_`{|}~]", text)
        words = [word.lower() for word in words if word != '\n' and word != ' ' and word != '']
        if (self.detect_punctuation):
            words = [word for word in words if word.strip() not in punctuation]
        if (self.detect_stop_words):
            words = [word for word in words if word.strip() not in self.stopwords_list]

        analyzed_words = []
        for word in words:
            lemmatized_word = self.pymorphy_analyzer.parse(word)[0].normal_form
            analyzed_words.append(lemmatized_word)

        return analyzed_words

#
# lemmatizer = PyMorphyLemmatizer(detect_punctuation=True, detect_stop_words=True)
# print(lemmatizer.lemmatize('По асфальту мимо цемента, Избегая зевак под аплодисменты. Обитатели спальных аррондисманов'))