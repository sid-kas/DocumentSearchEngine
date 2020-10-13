

def create_data_set():
    import pandas as pd

    data_path = "/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs.AI.tsv"

    df = pd.read_csv(data_path, sep="\t")

    print(df.head())

    abstracts = df["abstract"].tolist()

    with open("/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs_ai_train.txt", "w") as f:
        for abstract in abstracts[:-10]:
            f.writelines(abstract + "\n")

    with open("/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs_ai_test.txt", "w") as f:
        for abstract in abstracts[-10:]:
            f.writelines(abstract + "\n")

def predict():
    import logging
    from simpletransformers.language_generation import LanguageGenerationModel

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 256}, use_cuda=False)

    prompts_ = [
        "Despite the recent successes of deep learning, such models are still far from some human abilities like learning from few examples, reasoning and explaining decisions. In this paper, we focus on organ annotation in medical images and we introduce a reasoning framework that is based on learning fuzzy relations on a small dataset for generating explanations.",
        "There is a growing interest and literature on intrinsic motivations and open-ended learning in both cognitive robotics and machine learning on one side, and in psychology and neuroscience on the other. This paper aims to review some relevant contributions from the two literature threads and to draw links between them.",
        "Recent success of pre-trained language models (LMs) has spurred widespread interest in the language capabilities that they possess. However, efforts to understand whether LM representations are useful for symbolic reasoning tasks have been limited and scattered.",
        "Many theories, based on neuroscientific and psychological empirical evidence and on computational concepts, have been elaborated to explain the emergence of consciousness in the central nervous system. These theories propose key fundamental mechanisms to explain consciousness, but they only partially connect such mechanisms to the possible functional and adaptive role of consciousness.",
    ]

    prompts = [
        "Despite the recent successes of deep learning",
        "Learning in both cognitive and ",
        "I do not understand"
    ]

    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = '.'.join(generated[0].split('.')[:-1]) + '.'
        print(generated)
        print("----------------------------------------------------------------------")


from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "train_batch_size": 16,
    "num_train_epochs": 3,
    "mlm": False,
}

model = LanguageModelingModel('gpt2', 'gpt2', args=train_args, use_cuda=False)

model.train_model("/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs_ai_train.txt", eval_file="/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs_ai_test.txt")

model.eval_model("/mnt/InternalStorage/sidkas/DocumentSearchEngine/assets/sample_docs_eng/cs_ai_test.txt")