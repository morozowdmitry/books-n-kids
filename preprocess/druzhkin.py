import pymorphy2
import re
from preprocess.lemmatize import PyMorphyLemmatizer
from collections import Counter
from string import punctuation
from deeppavlov.models.tokenizers.ru_sent_tokenizer import ru_sent_tokenize
from preprocess.tokenize import Tokenizer
import numpy as np
from pathlib import Path


class DruzhkinAnalyzer(object):
    def __init__(self):
        base_path = Path(__file__).parent
        self.preprocess_folder = (base_path / "../preprocess").resolve()

        # Граммемы из диплома Дружкина. Стр 49
        # http://opencorpora.org/dict.php?act=gram
        self.grammems = ['ADJF', 'ADJS', 'ADVB', 'CONJ', 'PRCL', 'NOUN', 'NPRO',
                         'VERB', 'past', 'accs', 'acc2', 'datv', 'loct', "loc1",
                         'gent', 'gen1', 'ablt', 'plur', 'GRND', 'indc',
                         'impr', 'PRTF', "PRTS", '1per', '2per', 'femn',
                         'neut', 'perf', 'actv', 'pssv', 'inan', 'intr',
                         'nomn', '3per', "masc"]
        self.word_ends = ['-то', 'ак', 'ал', 'в', 'вот', 'все',
                          'гда', 'го', 'дь', 'ел', 'ие', 'ием',
                          'ии', 'ия', 'й', 'л', 'лся', 'не', 'ние',
                          'нии', 'ний', 'нию', 'ния', 'но', 'ной',
                          'ные', 'ных', 'о', 'ого', 'ой', 'он',
                          'се', 'сти', 'сь', 'так', 'ти', 'то', 'у',
                          'х', 'ции', 'ше', 'шь', 'ые', 'ыло', 'ых',
                          'ь', 'это', 'я', 'е', 'ы', 'ты', 'ый', 'ать',
                          'вие', 'еть', 'ить', 'кий', 'ный', 'сть',
                          'тво', 'тья', 'ция', 'что', 'щий', 'ание',
                          'атья', 'ация', 'деть', 'ение', 'нный', 'огда',
                          'ость', 'рить', 'ский', 'ство', 'твие', 'ьный',
                          'ящий', 'вание', 'дение', 'енный', 'еский', 'жение',
                          'знать', 'идеть', 'ление', 'льный', 'нение', 'ность',
                          'овать', 'орить', 'оящий', 'ствие', 'татья', 'шение']

        # ['-то', 'ак', 'ал', 'в', 'вот', 'все',
        #                   'гда', 'го', 'дь', 'ел', 'ие', 'ием', 'ии', 'ия', 'й', 'л', 'лся',
        #                   'не', 'ние', 'нии', 'ний', 'нию', 'ния', 'но', 'ной', 'ные', 'ных',
        #                   'о', 'ого', 'ой', 'он', 'се', 'сти', 'сь', 'так', 'ти', 'то', 'у',
        #                   'х', 'ции', 'ше', 'шь', 'ые', 'ыло', 'ых', 'ь', 'это', 'я',
        #                   'е', 'й', 'о', 'у', 'ы', 'ак', 'ие',
        #                   'ия', 'не', 'но', 'се', 'ти', 'то', 'ты', 'ый',
        #                   'ать', 'вие', 'вот', 'все', 'гда', 'еть', 'ить', 'кий',
        #                   'ние', 'ный', 'сть', 'так', 'тво', 'тья', 'ция', 'что',
        #                   'щий', 'ание', 'атья', 'ация', 'деть', 'ение', 'нный', 'огда',
        #                   'ость', 'рить', 'ский', 'ство', 'твие', 'ьный', 'ящий',
        #                   'вание', 'дение', 'енный', 'еский', 'жение', 'знать', 'идеть',
        #                   'ление', 'льный', 'нение', 'ность', 'овать', 'орить', 'оящий',
        #                   'ствие', 'татья', 'шение']

        self.tops = ['top_800_nouns', 'rk_100_1000_nouns', 'top_700_nouns', 'top_600_nouns',
                     'rk_100_1000_verbs', 'top_100_verbs', 'top_1000', 'top_300', 'top_500',
                     'rk_100_2500', 'top_900', 'top_700', 'top_200', 'top_800', 'top_400',
                     'top_100', 'top_600']

        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatizer = PyMorphyLemmatizer()

    def parse_word(self, word):
        return self.morph.parse(word)[0]

    def _get_num_of_words_from_list(self, words, list1):
        count = 0
        for word in words:
            if word in list1:
                count += 1
        return count

    def analyze(self, text):
        words = re.findall(r"[\w']+|[!\"#$%&'()*+,\-./:;<=>?@[/\]^_`{|}~]", text)
        words = [word.lower() for word in words if word != '\n' and word != ' ' and word != '']



        count_grammems = dict(zip(self.grammems, [0] * len(self.grammems)))
        count_ends = dict(zip(self.word_ends, [0] * len(self.word_ends)))
        for word in words:
            tag = self.parse_word(word).tag
            for g in self.grammems:
                if g in tag:
                    count_grammems[g] += 1
        for word in words:
            for end in self.word_ends:
                if word.endswith(end):
                    count_ends[end] += 1
        for key in count_grammems:
            count_grammems[key] /= len(words)
        for key in count_ends:
            count_ends[key] /= len(words)

        top_1000_nouns = [line.rstrip('\n') for line in
                          open(self.preprocess_folder / 'top_1000_nouns.txt', encoding='utf-8', mode='r')]
        top_1000_verbs = [line.rstrip('\n') for line in
                          open(self.preprocess_folder / 'top_1000_verbs.txt', encoding='utf-8', mode='r')]
        top_2500_words = [line.rstrip('\n') for line in
                          open(self.preprocess_folder / 'top_2500_words.txt', encoding='utf-8', mode='r')]
        words = self.lemmatizer.lemmatize(text)

        count_tops = {'top_800_nouns': self._get_num_of_words_from_list(words, top_1000_nouns[:800]),
                      'rk_100_1000_nouns': self._get_num_of_words_from_list(words, top_1000_nouns[100:]),
                      'top_700_nouns': self._get_num_of_words_from_list(words, top_1000_nouns[:700]),
                      'top_600_nouns': self._get_num_of_words_from_list(words, top_1000_nouns[:600]),
                      'rk_100_1000_verbs': self._get_num_of_words_from_list(words, top_1000_verbs[100:]),
                      'top_100_verbs': self._get_num_of_words_from_list(words, top_1000_verbs[:100]),
                      'top_1000': self._get_num_of_words_from_list(words, top_2500_words[:1000]),
                      'top_300': self._get_num_of_words_from_list(words, top_2500_words[:300]),
                      'top_500': self._get_num_of_words_from_list(words, top_2500_words[:500]),
                      'rk_100_2500': self._get_num_of_words_from_list(words, top_2500_words[100:]),
                      'top_900': self._get_num_of_words_from_list(words, top_2500_words[:900]),
                      'top_700': self._get_num_of_words_from_list(words, top_2500_words[:700]),
                      'top_200': self._get_num_of_words_from_list(words, top_2500_words[:200]),
                      'top_800': self._get_num_of_words_from_list(words, top_2500_words[:800]),
                      'top_400': self._get_num_of_words_from_list(words, top_2500_words[:400]),
                      'top_100': self._get_num_of_words_from_list(words, top_2500_words[:100]),
                      'top_600': self._get_num_of_words_from_list(words, top_2500_words[:600])}

        if len(words) == 0:
            return np.zeros(147)

        for key in count_tops:
            count_tops[key] /= len(words)

        return np.array(list(count_grammems.values()) + list(count_ends.values()) + list(count_tops.values()))
        # return count_grammems, count_ends, count_tops


class SyntaxFeatures:
    def get_punctuation_count(self, text):
        count_punctuation = Counter()
        for sign in punctuation:
            cnt = text.count(sign)
            if cnt:
                count_punctuation[sign] = cnt
            else:
                count_punctuation[sign] = 0
        return count_punctuation

    def get_sentence_lengths(self, text):
        text = text.replace('\n', '')
        text = re.sub(' +', ' ', text)
        sentences = ru_sent_tokenize(text)
        for sent in sentences:
            if len(sent) < 1:
                sentences.remove(sent)
        sent_lengths = []

        for sent in sentences:
            sent_lengths.append(len(Tokenizer().tokenize(sent)))

        return sent_lengths

# analyzer = DruzhkinAnalyzer()
# # p = analyzer.parse_word('идет')
# text = 'РИА Новости - события в Москве, России и мире сегодня ...ria.ru Новости в России и мире, самая оперативная ' \
#        'информация: темы дня, обзоры, анализ. Фото и видео с места событий, инфографика, радиоэфир, ... '
# a, b, c = analyzer.analyze(text)
# print((list(a.values()) + list(b.values()) + list(c.values())))
# #
# print(b.values())
# print(c.values())
#
#
# import numpy as np
# Alice = ''
# with open('../books/school/Alice.txt', 'r', encoding='utf-8') as file:
#     Alice = file.read()
# sent_lengths = SyntaxFeatures().get_sentence_lengths(Alice)
# print('max', max(sent_lengths))
# print('average', np.average(sent_lengths))
#

#
# text = 'РИА Новости - события в Москве, России и мире сегодня ...ria.ru Новости в России и мире, самая оперативная ' \
#        'информация: темы дня, обзоры, анализ. Фото и видео с места событий, инфографика, радиоэфир, ... '
# print(SyntaxFeatures().get_punctuation_count(text))
