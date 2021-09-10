from predict.cnn import KFold_predict
from preprocess.druzhkin import DruzhkinAnalyzer
import pandas as pd
import numpy as np
import re
from deeppavlov.models.tokenizers.ru_sent_tokenizer import ru_sent_tokenize
sentence_slice = 70


def predict_text_class(text):
    text = text.replace('—', '')
    analyzer = DruzhkinAnalyzer()
    text = text.replace('\n', '')
    text = re.sub(' +', ' ', text)
    sentences = ru_sent_tokenize(text)
    if len(sentences) < 1:
        return 0, 0
    slices = []
    metas = []
    for num in np.arange(0, len(sentences), sentence_slice):
        text_slice = ' '.join(sentences[num:num + sentence_slice])
        meta = analyzer.analyze(text_slice)
        slices.append(text_slice)
        metas.append(meta)

    d = {'text': slices, 'metadata': metas}
    df = pd.DataFrame(d)
    predict = KFold_predict(df, 6)

    predict = np.sum(predict, axis=0)
    mean_pred = np.mean(predict, axis=0)
    idx = np.argmax(mean_pred, axis=0)
    return idx, mean_pred[idx]/np.sum(mean_pred)


if __name__ == "__main__":
    with open("test_text.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        # TODO Text exceptions (like empty text)
        test_pred = predict_text_class(text)
        print(test_pred)   # out: (class_idx, probability)
