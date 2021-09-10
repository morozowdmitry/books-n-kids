from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from string import punctuation


class Tokenizer:
    def tokenize(self, text, remove_punctuation=True):
        words = NLTKTokenizer().tokenizer(text)
        if remove_punctuation:
            for word in words:
                for sign in punctuation:
                    if sign in word and word in words:
                        words.remove(word)
        return words

# text = 'РИА Новости - события в Москве, России и мире сегодня ...ria.ru Новости в России и мире, самая оперативная ' \
#        'информация: темы дня, обзоры, анализ. Фото и видео с места событий, инфографика, радиоэфир, ... '
# print(Tokenizer().tokenize(text))
