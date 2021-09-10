import pandas as pd
import numpy as np
import os
import math

books_paths = ['tokenized_70_0',
               'tokenized_70_1',
               'tokenized_70_2']

sentence_slice = 70


def get_books_list():
    df_books = pd.DataFrame(columns=['filename', 'class'])
    for idx in range(len(books_paths)):
        for file_name in os.listdir(books_paths[idx]):
            append_data = dict(zip(df_books.columns, [file_name, idx]))
            df_books = df_books.append(append_data, ignore_index=True)
    return df_books


def get_books_by_percent(df, perc):
    new_df = pd.DataFrame(columns=['filename', 'class'])
    new_df = new_df.append(df[df['class'] == 0].sample(math.ceil(len(df[df['class'] == 0]) * perc)))
    new_df = new_df.append(df[df['class'] == 1].sample(math.ceil(len(df[df['class'] == 1]) * perc)))
    new_df = new_df.append(df[df['class'] == 2].sample(math.ceil(len(df[df['class'] == 2]) * perc)))
    return new_df


folder_name = 'tokenized_70_'
separator = '\\sep'


def prepare_books_dataframe(df_books):
    df = pd.DataFrame(columns=['bookname', 'metadata', 'text', 'class'])

    for index, row in df_books.iterrows():
        idx = row['class']
        file_name = row['filename']

        with open(os.path.join(folder_name + str(idx), file_name), 'r', encoding='utf-8') as text_file:
            text = text_file.read()
            parts = text.split(separator)
            metadata = np.loadtxt('analyzed_' + str(idx) + '/' + file_name)
            metadata = np.reshape(metadata, (-1, 147))
            for i in range(len(parts)):
                append_data = dict(zip(df.columns, [file_name, metadata[i], parts[i], idx]))
                df = df.append(append_data, ignore_index=True)

    df.fillna(0)
    return df


if __name__ == '__main__':
    df_books = get_books_list()

    df_books.to_csv('books_list.csv')
    print("TOTAL ALL BOOKS")
    print(df_books['class'].value_counts())

    valid_books = get_books_by_percent(df_books, 0.15)
    df_books = df_books.drop(valid_books.index)
    print("Val")
    print(valid_books['class'].value_counts(sort=False))

    test_books = get_books_by_percent(df_books, 0.15)
    df_books = df_books.drop(test_books.index)
    print("test")
    print(test_books['class'].value_counts(sort=False))

    print("train")
    print(df_books['class'].value_counts(sort=False))

    train_books = df_books
    train = prepare_books_dataframe(train_books)
    train.to_csv('train_texts_70.csv')

    test = prepare_books_dataframe(test_books)
    test.to_csv('test_texts_70.csv')

    valid = prepare_books_dataframe(valid_books)
    train.to_csv('valid_texts_70.csv')