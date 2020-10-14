import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME

class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(trainer, paths)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


class GPT2Train(object):

    def __init__(self,text_docs_dir, model_save_dir='./model_gpt2_test/'):
        self.text_docs_dir = text_docs_dir
        self.tokenized_data_dir = model_save_dir+'/tokenized_data'
        self.model_save_dir = model_save_dir
        self.model = None; self.dataset = None; self.tokenizer = None

    def tokenize(self):
        # the folder 'text' contains all the files
        paths = [str(x) for x in Path(self.text_docs_dir).glob("**/*.txt")]
        tokenizer = BPE_token()
        # train the tokenizer model
        tokenizer.bpe_train(paths)
        # saving the tokenized data in our specified folder 
        tokenizer.save_tokenizer(self.tokenized_data_dir)

    def create_dataset(self, block_size = 100, BATCH_SIZE = 16, BUFFER_SIZE = 1000):
        tokenizer = self._get_tokenizer()
        single_string = ''
        paths = [str(x) for x in Path(self.text_docs_dir).glob("**/*.txt")]
        for filename in paths:
            with open(filename, "r", encoding='utf-8') as f:
                x = f.read()
            single_string += x + tokenizer.eos_token
        string_tokenized = tokenizer.encode(single_string)

        examples = []
        for i in range(0, len(string_tokenized) - block_size + 1, block_size):
            examples.append(string_tokenized[i:i + block_size])

        inputs, labels = [], []
        for ex in examples:
            inputs.append(ex[:-1])
            labels.append(ex[1:])

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        self.dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        return self.dataset

    def model_init(self):
        tokenizer = self._get_tokenizer()
        # creating the configurations from which the model can be made
        config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
        )
        # creating the model
        self.model = TFGPT2LMHeadModel(config)
        return self.model

    def train(self, num_epoch = 10):
        # defining our optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        # definining our loss function
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # defining our metric which we want to observe
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        # compiling the model
        self.model.compile(optimizer=optimizer, loss=[loss, *[None] * self.model.config.n_layer], metrics=[metric])
        history = self.model.fit(self.dataset, epochs=num_epoch)
        print(history)
        self.save_model()

    def _get_tokenizer(self):
        from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
        # loading tokenizer from the saved model path
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenized_data_dir)
        self.tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        })
        return self.tokenizer

   

    def save_model(self):
        # creating directory if it is not present
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = os.path.join(self.model_save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.model_save_dir, CONFIG_NAME)
       
        # save model and model configs
        self.model.save_pretrained(self.model_save_dir)
        model_to_save.config.to_json_file(output_config_file)
       
        # save tokenizer
        self.tokenizer.save_pretrained(self.model_save_dir)

 


class GPT2Predict:
    def __init__(self, model_dir):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = TFGPT2LMHeadModel.from_pretrained(model_dir)

    def predict(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='tf')
        # getting out output
        beam_output = self.model.generate(
            input_ids,
            max_length = 50,
            num_beams = 5,
            temperature = 0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=5
        )

        return beam_output


def train_sample_docs():

    eng_txt_doc = "C:\\Users\\nwire\\project_ks_protokoll\\DocumentSearchEngine\\assets\\sample_docs_eng"
    gpt2_trainer = GPT2Train(text_docs_dir=eng_txt_doc, model_save_dir='./model_gpt2_eng_sample/')

    # gpt2_trainer.tokenize()
    gpt2_trainer.create_dataset(block_size=100, BATCH_SIZE=12, BUFFER_SIZE=1000)
    gpt2_trainer.model_init()

    gpt2_trainer.train(num_epoch=10)




train_sample_docs()
