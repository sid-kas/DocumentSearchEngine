"""
Example script to create elasticsearch documents.
"""
import argparse
import json
from tqdm import tqdm

import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(output_fmt='list',port=5555, port_out=5556)

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import re
import base64
import numpy as np

dfloat32 = np.dtype('>f4')

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=dfloat32).tolist()

def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dfloat32)).decode("utf-8")
    return base64_str

def load_textfile(path):
    # load data
    with open(path,'r') as f:
        file_input=f.read()
    return file_input

def clean_text(text):
    sents = sent_tokenize(text,language='english')
    return sents

def embed(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc for doc in batch_docs])
        for emb in embeddings:
            yield emb

def create_docs(index_name, path, doc_id, title):
    docs = []
    data_set = clean_text(load_textfile(path))
    print("cleaned text!!")
    writer = open(args.save, 'w')
    pbar = tqdm(total=len(data_set))
    for item, emb in tqdm(zip(data_set, embed(data_set))):
        doc = {
            '_op_type': 'index',
            '_index': index_name,
            'doc_id': doc_id,
            'title': title,
            'text': item,
            'text_vector': emb
        }
        pbar.update()
        writer.write(json.dumps(doc) + '\n')
    writer.close()
    pbar.close()

def main(args):
    path = str(args.data)
    create_docs(args.index_name, args.data, doc_id=path, title=path.split('.')[0].split('/')[-1])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='docsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)