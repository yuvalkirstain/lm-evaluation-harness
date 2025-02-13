import os

import datasets
from math import exp

from datasets import load_dataset, DatasetDict
from transformers.data.metrics import squad_metrics

from lm_eval.base import rf
from lm_eval.metrics import f1_score, mean
from .common import HFTask
from functools import partial

from ..utils import sh


class NQOpen(HFTask):
    DATASET_PATH = "nq_open"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.data["train"]

    def validation_docs(self):
        return self.data["validation"]

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return 'Question: ' + doc['question'] + '\n\n' + 'Answer:'

    def doc_to_target(self, doc):
        answer_list = doc['answer']
        answer = answer_list[0]
        return " " + answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n'])
        return continuation,

    @staticmethod
    def compute_scores(gold_list, pred):
        # tests for exact match and on the normalised answer (compute_exact)
        # test for overlap (compute_f1)
        em = max(squad_metrics.compute_exact(a, pred) for a in gold_list)
        f1 = max(squad_metrics.compute_f1(a, pred) for a in gold_list)

        return {'em': em * 100, 'f1': f1 * 100}

    def process_results(self, doc, results):
        """Take a single document and the LM results_old and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results_old of the requests created in construct_requests.
        """
        continuation, = results

        return self.compute_scores(doc['answer'], continuation)

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            'exact': True,  # Exact match (the normalized answer exactly match the gold answer)
            'f1': True,  # The F-score of predicted tokens versus the gold answer
        }


class NQOpenNoOverlap(NQOpen):
    def __init__(self):
        self.data_dir = f'data/nq_open/'
        self.test_path = f"{self.data_dir}/nq-test.qa.no_overlap.json"
        super().__init__()

    def download(self):
        super().download()
        sh(f"""
            mkdir -p {self.data_dir}
            wget https://dl.fbaipublicfiles.com/when_do_billions/nq-test.qa.no_overlap.jsonl -O {self.test_path}
            """)
        self.data["test"] = load_dataset('json', data_files=self.test_path)["train"]

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True


class WebQsOurs(NQOpen):
    DATASET_PATH = "web_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        out_doc = {'url': doc["url"],
                   'question': doc["question"],
                   'answer': doc["answers"]}
        return out_doc

    def training_docs(self):
        # Cache training for faster few-shot.
        # If data is too large to fit in memory, override this method.
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._convert_standard, self.data["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._convert_standard, self.data["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._convert_standard, self.data["test"])


class TriviaQAOurs(NQOpen):
    def __init__(self):
        self.data_dir = f'data/triviaqa/'
        self.train_path = f"{self.data_dir}/triviaqa.train-train.json"
        self.test_path = f"{self.data_dir}/triviaqa.test.jsonl"
        super().__init__()

    def download(self):
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            sh(f"""
                mkdir -p {self.data_dir}
                wget https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.train-train.jsonl -O {self.train_path}
                wget https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.test.jsonl -O {self.test_path}
                """)
        data_files = {"train": self.train_path, "test": self.test_path}
        self.data = load_dataset('json', data_files=data_files)

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True
