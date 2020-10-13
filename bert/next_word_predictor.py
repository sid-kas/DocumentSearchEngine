import os, traceback
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM, Bert


def load_model(model_name):
  try:
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertForMaskedLM.from_pretrained(model_name).eval()
    return bert_tokenizer,bert_model
  except Exception as e:
    pass

def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
  # if <mask> is the last token, append a "." so that models dont predict punctuation.
  print(text_sentence)
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

  input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
  mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx


def get_all_predictions(bert_tokenizer, bert_model,input_text, top_clean=5):
  input_ids, mask_idx = encode(bert_tokenizer, input_text)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_clean).indices.tolist(), top_clean)
  return bert


def get_prediction_eos(bert_tokenizer, bert_model, input_text, n_words):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(bert_tokenizer, bert_model, input_text, top_clean=int(n_words))
    return res
  except Exception as error:
    pass

def main():
  model_name = ['bert-base-uncased', 'af-ai-center/bert-base-swedish-uncased', 'KB/bert-base-swedish-cased', 'bert-base-multilingual-cased']
  bert_tokenizer, bert_model = load_model(model_name[2])
  try: 
    while True:
      query = input("Enter word or a sentence: ")
      predictions = get_prediction_eos(bert_tokenizer, bert_model, query, n_words=5)
      print(predictions.splitlines())
  except KeyboardInterrupt as e:
      print(e)
      exit(0)
  except BaseException as e:
      traceback.print_exc()
      print(e)



main()