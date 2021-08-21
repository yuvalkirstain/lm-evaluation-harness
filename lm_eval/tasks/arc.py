import json
import os

from lm_eval.base import MultipleChoiceTask
from . common import HFTask
from ..utils import sh


class ARCEasy(HFTask, MultipleChoiceTask):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        out_doc = {
            "id": doc["id"],
            "query": "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]


class ARCEasyCls(ARCEasy):
    def _convert_standard(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        options_str = "\n".join([f"{letter}: {option}" for letter, option in zip(doc["choices"]["label"], doc["choices"]["text"])])
        out_doc = {
            "id": doc["id"],
            "query": f"\nOptions: {options_str}" + "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"]["label"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc


class ARCEasyExtractive(ARCEasy):
    def _convert_standard(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        options_str = "\n".join([f"{letter}: {option}" for letter, option in zip(doc["choices"]["label"], doc["choices"]["text"])])
        out_doc = {
            "id": doc["id"],
            "query": f"Question: {doc['question']}\nCandidates:\n{options_str}\nAnswer:",
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc


class ARCEasyIR(ARCEasyExtractive):
    def __init__(self):
        self.data_dir = "data/arc_easy"
        self.file_name = "ARC-OBQA-RegLivEnv-IR10V8"
        super().__init__()

    def download(self):
        if not os.path.exists(self.data_dir):
            sh(f"""
mkdir -p {self.data_dir}
wget https://raw.githubusercontent.com/allenai/unifiedqa/master/files/arc-with-ir/{self.file_name}.zip -O {self.data_dir}/{self.file_name}.zip
unzip {self.data_dir}/{self.file_name}.zip -d {self.data_dir}
""")

    def read_jsonl(self, path):
        split = []
        with open(path) as f:
            for line in f:
                data_point = json.loads(line)
                if not data_point["id"].startswith("ARCE"):
                    continue
                split.append(self._convert_standard(data_point))
        return split

    def training_docs(self):
        path = f'{self.data_dir}/{self.file_name}/train.jsonl'
        return self.read_jsonl(path)

    def validation_docs(self):
        path = f'{self.data_dir}/{self.file_name}/dev.jsonl'
        return self.read_jsonl(path)

    def test_docs(self):
        path = f'{self.data_dir}/{self.file_name}/test.jsonl'
        return self.read_jsonl(path)

    def _convert_standard(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        options_str = "\n".join(
            [f"{candidate['label']}: {candidate['text']}" for candidate in doc["question"]["choices"]])
        out_doc = {
            "id": doc["id"],
            "query": f"Background: {doc['para']}\nQuestion: {doc['question']['stem']}\nCandidates:\n{options_str}\nAnswer:",
            "choices": [choice["text"] for choice in doc["question"]["choices"]],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
