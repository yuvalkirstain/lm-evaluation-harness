from lm_eval.base import MultipleChoiceTask
from . common import HFTask


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




class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
