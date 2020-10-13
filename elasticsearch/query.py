from elasticsearch import Elasticsearch
from bert_serving.client import BertClient
import json

import base64
import numpy as np

dfloat32 = np.dtype('>f4')

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=dfloat32).tolist()

def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dfloat32)).decode("utf-8")
    return base64_str

SEARCH_SIZE = 10
INDEX_NAME = "docsearch"
def analyzer():
    bc = BertClient(output_fmt='list',port=5555, port_out=5556)
    client = Elasticsearch("localhost", port=9200)

    query = input("Enter query: ")
    query_vector = bc.encode([query])[0]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    simple_query = {
        "prefix": {
            "text": {
                "value": query
            }
        }
    }
    suggestions = {
        "text" : query,
        "simple_phrase": {
            "phrase": {
                "field": "text",
                # "size": 1,
                # "gram_size": 3,
                "direct_generator": [ {
                    "field": "text",
                    "suggest_mode": "missing",
                    "sort":"frequency"
                },
                {
                    "field" : "text",
                    "suggest_mode" : "always"
                },
                {
                    "field" : "text",
                    "suggest_mode" : "popular"
                } ],
        }
        }
        
    }

    mlt_query = {
        "more_like_this": {
                "fields": ["text"],
                "like": query,
                "min_term_freq": 1,
                "max_query_terms": 50,
                "min_doc_freq": 1
        }
    }

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": mlt_query,
            "suggest": suggestions,
            "_source": {"includes": ["title", "text", "line"]}
        }
    )
    print(query)
    print(json.dumps(response, indent=4, sort_keys=True))
    # return jsonify(response)


try:
    while True:
       analyzer()
except KeyboardInterrupt as e:
    print(e)
    exit(0)
except BaseException as e:
    print(e)


