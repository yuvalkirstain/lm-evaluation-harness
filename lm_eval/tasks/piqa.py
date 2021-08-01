import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from ..metrics import mean
from .common import HFTask


class PiQA(HFTask, MultipleChoiceTask):
    DATASET_PATH = "piqa"
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
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"


class PiQACls(PiQA):
    def _convert_standard(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": ["A", "B"],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        text = "A: " + doc["choices"][0] + "\nB: " + doc["choices"][1] + "\nQuestion: " + doc["goal"] + "\nAnswer:"
        return text


class PiQAExtractive(PiQA):
    def _convert_standard(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        text = f"Question: {doc['goal']}\nCandidates:\nA: {doc['choices'][0]}\nB: {doc['choices'][1]}\nAnswer:"
        return text

    def construct_requests(self, doc, ctx):
        choices_prefix = ""
        for w1, w2 in zip(doc["choices"][0].split(), doc["choices"][1].split()):
            if w1 == w2:
                choices_prefix += f" {w1}"
                continue
            break

        lls = [
            rf.loglikelihood(ctx, choices_prefix + f" {w1}")[0],
            rf.loglikelihood(ctx, choices_prefix + f" {w2}")[0],
        ]

        return lls