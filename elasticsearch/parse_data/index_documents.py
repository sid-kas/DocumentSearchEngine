"""
Example script to index elasticsearch documents.
"""
import argparse
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def create_index(index_file, index_name):
    client = Elasticsearch()
    client.indices.delete(index=index_name, ignore=[404])
    with open(index_file) as f:
        source = f.read().strip()
        client.indices.create(index=index_name, body=source)

def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(args):
    create_index(args.index_file, args.index_name)
    client = Elasticsearch()
    docs = load_dataset(args.data)
    bulk(client, docs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indexing elasticsearch documents.')
    parser.add_argument('--data', default='documents.jsonl', help='Elasticsearch documents.')
    parser.add_argument('--index_file', default='index.json', help='Elasticsearch index file.')
    parser.add_argument('--index_name', default='docsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)