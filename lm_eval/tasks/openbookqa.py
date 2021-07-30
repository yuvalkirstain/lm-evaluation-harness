from lm_eval.base import MultipleChoiceTask
from .common import HFTask


class OpenBookQA(HFTask, MultipleChoiceTask):
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        out_doc = {
            "id": doc["id"],
            "query": doc["question_stem"],
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]


class OpenBookQAUnifiedQA(OpenBookQA):
    def _convert_standard(self, doc):
        letters = ["A", "B", "C", "D"]
        assert len(letters) == len(doc['choices']['text'])
        candidate_line = " ".join([f"({letters[i]}) {doc['choices']['text'][i]}" for i in range(len(letters))])
        out_doc = {
            "id": doc["id"],
            "query": f'{doc["question_stem"]}\n{candidate_line}\n',
            "choices": doc["choices"]["text"],
            "gold": letters.index(doc["answerKey"].strip()),
        }
        return out_doc
