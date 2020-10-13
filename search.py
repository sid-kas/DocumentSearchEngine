from elasticsearch import Elasticsearch
import os

es = Elasticsearch("localhost", port=9200)

def request_search():
    try:
        while True:
            result = []
            search_term = input("Enter query: ")
            res = es.search(
            index='fscrawler_job',
            body={
            "query" : {"match": {"content": search_term}},
            "highlight" : {"pre_tags" : ["<b>"] , "post_tags" : ["</b>"], "fields" : {"content":{}}}})
            res['ST']=search_term
            for hit in res['hits']['hits']:
                hit['good_summary']='….'.join(hit['highlight']['content'][1:])
                result.append(hit)
            print(len(result))
            for r in result:
                print(r['good_summary'].encode())
    except KeyboardInterrupt as e:
        print(e)
        exit(0)
    except BaseException as e:
        print(e)


request_search()

    