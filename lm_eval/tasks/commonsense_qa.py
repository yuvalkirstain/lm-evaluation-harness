import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from ..metrics import mean
from .common import HFTask


class CommonsenseQA(HFTask, MultipleChoiceTask):
    DATASET_PATH = "commonsense_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def _convert_standard(self, doc):
        out_doc = {
            "question": doc["question"],
            "labels": doc["choices"]["label"],
            "choices": doc["choices"]["text"],
            "gold": doc["choices"]["label"].index(doc["answerKey"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"


class xCommonsenseQAExtractive(CommonsenseQA):

    def doc_to_text(self, doc):
        candidates = '\n'.join([f"{label}: {choice}" for label, choice in zip(doc["labels"], doc["choices"])])
        return f"Question: {doc['question']}\nCandidates:\n{candidates}\nAnswer:"
