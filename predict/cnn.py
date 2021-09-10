import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import torch
from torchtext.legacy import data
from string import punctuation
from torchtext.vocab import Vectors
import time
import os
import pickle
from sklearn.model_selection import KFold
import re
# from predict.baseline_model import get_books_list, prepare_books_dataframe
import spacy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
from pathlib import Path

base_path = Path(__file__).parent
predict_path = (base_path / "../predict").resolve()
nlp = spacy.load("ru_core_news_lg")
nltk.download('stopwords')
stopwords = stopwords.words("russian")
vectors = Vectors(name=predict_path / 'multilingual_embeddings.ru', cache='./')


def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


def remove_stopwords(text):
    res = []
    for w in text:
        word = w.lower()
        if word not in stopwords and word not in punctuation and '\n' not in word and not word.isdigit():
            res.append(word)
    return res


def string_arr_to_list(str_arr):
    str_arr = re.sub(' +', ' ', str_arr)
    str_arr = re.sub('\n', '', str_arr)
    str_arr = re.sub('0\. ', '0.0 ', str_arr)
    str_arr = re.sub('e(-|\+)0\.0', '', str_arr)

    res = [float(s) for s in str_arr[1:-2].split(' ')]
    return res


# device = torch.cuda.device(2)
# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='ru_core_news_lg',
                  preprocessing=remove_stopwords)
LABEL = data.LabelField()
METADATA = data.Field(sequential=False, use_vocab=False, dtype=torch.float32,
                      preprocessing=string_arr_to_list)

fields = [('text', TEXT), ('label', LABEL), ('metadata', METADATA)]


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters + 147, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, metadata):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat_1 = torch.cat(pooled, dim=1)
        cat = self.dropout(torch.cat((cat_1, metadata), dim=1))
        return self.fc(cat)


def generate_best_model(df_train, df_valid):
    tsv_train = pd.DataFrame()
    tsv_train['text'] = df_train['text']
    tsv_train['label'] = df_train['class']
    tsv_train['metadata'] = df_train['metadata']
    tsv_train.to_csv('train.tsv', sep='\t', index=False)

    tsv_valid = pd.DataFrame()
    tsv_valid['text'] = df_valid['text']
    tsv_valid['label'] = df_valid['class']
    tsv_valid['metadata'] = df_valid['metadata']
    tsv_valid.to_csv('valid.tsv', sep='\t', index=False)

    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_data = data.TabularDataset(
        path='train.tsv',
        format='tsv',
        fields=fields,
        skip_header=True
    )
    valid_data = data.TabularDataset(
        path='valid.tsv',
        format='tsv',
        fields=fields,
        skip_header=True
    )

    MAX_VOCAB_SIZE = 25_000
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=vectors,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    BATCH_SIZE = 32
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        device=device,
        sort=False)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    def categorical_accuracy(preds, y):
        top_pred = preds.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.train()

        for batch in tqdm(iterator):
            optimizer.zero_grad()
            predictions = model(batch.text, batch.metadata)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text, batch.metadata)
                loss = criterion(predictions, batch.label)
                acc = categorical_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = 10
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    model.load_state_dict(torch.load('model.pt'))
    return model, TEXT.vocab, LABEL.vocab.itos


def KFold_proprotional(dfs_idx, n):
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    idx_books = [list(kf.split(dfs_idx[class_idx])) for class_idx in range(3)]
    return idx_books


def do_KFold_CNNs(dfs_by_idx, n_split):
    idx_folds = KFold_proprotional(dfs_by_idx, n_split)

    for i in range(n_split - 1):
        valid_books = dfs_by_idx[0].iloc[idx_folds[0][i][1]]
        valid_books = valid_books.append(dfs_by_idx[1].iloc[idx_folds[1][i][1]])
        valid_books = valid_books.append(dfs_by_idx[2].iloc[idx_folds[2][i][1]])

        train_books = dfs_by_idx[0].iloc[idx_folds[0][i][0]]
        train_books = train_books.append(dfs_by_idx[1].iloc[idx_folds[1][i][0]])
        train_books = train_books.append(dfs_by_idx[2].iloc[idx_folds[2][i][0]])

        valid = prepare_books_dataframe(valid_books)
        train = prepare_books_dataframe(train_books)

        print('working on model ' + str(i))
        model, vocab, label_vocab = generate_best_model(train, valid)
        torch.save(model.state_dict(), 'cnn_models/model_' + str(i) + '.pt')
        save_vocab(vocab, 'vocab_' + str(i))
        label_vocab = np.array([int(l) for l in label_vocab])
        np.savetxt('label_vocab_' + str(i), label_vocab)


def KFold_predict(df_test, n_models):
    texts = df_test['text']
    metadata = df_test['metadata']
    INPUT_DIM = 25002
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 3
    DROPOUT = 0.5

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model = model.to(device)

    predicts = []
    for i in tqdm(range(n_models)):
        model_file = os.path.join(predict_path / 'cnn_models', 'model_' + str(i) + '.pt')
        model.load_state_dict(torch.load(model_file))
        label_vocab = np.loadtxt(predict_path / ('label_vocabs/label_vocab_' + str(i)))
        label_vocab = [int(l) for l in label_vocab]
        # print(label_vocab)

        with open(predict_path / ("vocabs/vocab_" + str(i)), 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
            model_predicts = []
            for text, meta in zip(texts, metadata):
                text_predict = predict_classes(model, vocab, text, meta)
                temp = np.zeros(3)

                for l, i in zip(label_vocab, range(len(label_vocab))):
                    temp[l] = text_predict[0][i]

                text_predict[0] = temp
                model_predicts.append(text_predict)
            predicts.append(model_predicts)

    predicts = np.array(predicts)
    predicts = np.mean(predicts, axis=2)
    predicts = np.swapaxes(predicts, 0, 1)
    return predicts


def predict_classes(model, vocab, sentence, metadata, min_len=5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    tokenized = remove_stopwords(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    meta_tensor = torch.FloatTensor(metadata.astype(float)).to(device).unsqueeze(0)

    preds = model(tensor, meta_tensor)
    preds = torch.sigmoid(preds)
    return preds.cpu().detach().numpy()


if __name__ == '__main__':
    n_split = 7
    n_models = n_split - 1
    df_books = get_books_list()
    dfs_by_idx = [df_books[df_books['class'] == class_idx].reset_index(drop=True) for class_idx in range(3)]

    idx_folds = KFold_proprotional(dfs_by_idx, n_split)
    test_books = dfs_by_idx[0].iloc[idx_folds[0][n_models][1]]
    test_books = test_books.append(dfs_by_idx[1].iloc[idx_folds[1][n_models][1]])
    test_books = test_books.append(dfs_by_idx[2].iloc[idx_folds[2][n_models][1]])
    test_books = test_books.append(dfs_by_idx[0].iloc[idx_folds[0][n_models][0]])
    test_books = test_books.append(dfs_by_idx[1].iloc[idx_folds[1][n_models][0]])
    test_books = test_books.append(dfs_by_idx[2].iloc[idx_folds[2][n_models][0]])

    test = prepare_books_dataframe(test_books)
    print(test['class'].to_numpy(dtype=int))

    do_KFold_CNNs(dfs_by_idx, n_split)
    test_pred = KFold_predict(test, n_models)  # [texts_num, n_models, 3]

    votes = np.argmax(test_pred, axis=2)  # [texts_num, n_models]
    vote_pred = [np.argmax(np.bincount(text_votes)) for text_votes in votes]
    print("Acc Hard voting : " + str(accuracy_score(vote_pred, test['class'].to_numpy(dtype=int))))

    mean_pred = np.mean(test_pred, axis=1)
    mean_pred = np.argmax(mean_pred, axis=1)
    print("Acc Soft voting : " + str(accuracy_score(mean_pred, test['class'].to_numpy(dtype=int))))

    print("Each model:")
    test_each = np.swapaxes(test_pred, 0, 1)
    test_each = np.argmax(test_each, axis=2)
    for model_idx in range(n_models):
        print("Acc " + str(model_idx) + " : " + str(
            accuracy_score(test_each[model_idx], test['class'].to_numpy(dtype=int))))
