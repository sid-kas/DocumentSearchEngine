# https://github.com/Hironsan/bertsearch
# https://huggingface.co/transformers/model_doc/xlmroberta.html

# https://github.com/huggingface/transformers
from transformers import BertTokenizer, TFBertModel

## https://github.com/Kungbib/swedish-bert-models
# tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
# model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
# model = TFAutoModel.from_pretrained('KB/bert-base-swedish-cased')

# https://github.com/af-ai-center/SweBERT OR https://github.com/af-ai-center/bert
pretrained_model_name = 'af-ai-center/bert-base-swedish-uncased'

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

model = TFBertModel.from_pretrained(pretrained_model_name)




example = 'Jag är ett barn, och det här är mitt hem. Alltså är det ett barnhem!'