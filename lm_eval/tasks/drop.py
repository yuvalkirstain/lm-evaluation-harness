import json
import numpy as np
import re
import string
from best_download import download_file
from scipy.optimize import linear_sum_assignment
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from pathlib import Path
from zipfile import ZipFile

"""
Acknowledgement: This implementation is based on the official evaluation for `DROP`:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
"""


class DROP(Task):
    DATASET_PATH = Path("data/drop")

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH)
        url = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
        checksum = "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"
        zip_path = self.DATASET_PATH / "drop_dataset.zip"
        download_file(url, str(zip_path), checksum)
        with ZipFile(zip_path, "r") as zip:
            zip.extractall(self.DATASET_PATH)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def _load_docs(self, docs):
        for doc in docs:
            for qa in doc["qa_pairs"]:
                yield {
                    "id": qa["query_id"],
                    "passage": doc["passage"],
                    "question": qa["question"],
                    "answers": self.get_answers(qa["answer"]),
                }

    @classmethod
    def get_answers(cls, answers):
        # NOTE: We wrap every non-`list` answer into a list for uniformity.
        if answers["number"] != "":
            return [str(answers["number"])]
        if answers["spans"] != []:
            return answers["spans"]
        return [" ".join([answers["date"]["day"],
                          answers["date"]["month"],
                          answers["date"]["year"]]).strip()]

    def training_docs(self):
        docs = json.load(open(self.DATASET_PATH / "drop_dataset" / "drop_dataset_train.json"))
        return self._load_docs([docs[k] for k in docs.keys()])

    def validation_docs(self):
        docs = json.load(open(self.DATASET_PATH / "drop_dataset" / "drop_dataset_dev.json"))
        return self._load_docs([docs[k] for k in docs.keys()])

    def doc_to_text(self, doc):
        return f"Passage: {doc['passage']}\nQuestion: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + ", ".join(doc["answers"])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        conts = []
        for _ in doc["answers"]:
            conts.append(rf.greedy_until(ctx, ["."]))
        return conts

    def process_results(self, doc, results):
        """Take a single document and the LM results_old and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results_old of the requests created in construct_requests.
        """
        preds, golds = results, doc["answers"]
        exact_match, f1_score = self.get_metrics(preds, golds)
        return {
            "em": exact_match,
            "f1": f1_score
        }

    def get_metrics(self, preds, golds):
        exact_match = self._exact_match(preds, golds) * 100
        f1_score = self._f1_score(preds, golds) * 100
        return exact_match, f1_score

    def _exact_match(self, preds, golds):
        """ Returns the exact match of normalized gold answers and predictions. """
        normalized_preds = [self._normalize(pred) for pred in preds]
        normalized_golds = [self._normalize(gold) for gold in golds]
        is_equal_sets = set(normalized_preds) == set(normalized_golds)
        is_equal_length = len(normalized_preds) == len(normalized_golds)
        return int(is_equal_sets and is_equal_length)

    def _f1_score(self, preds, golds):
        """Returns the average F1-score over normalized gold answers and predictions.
        From Section 5 of Dua et al. "DROP:...":
        "When an answer has multiple spans, we first perform a one-to-one
        alignment greedily based on bag-of-word overlap on the set of spans
        and then compute average F1 over each span."
        """
        pred_bags = self._answer_to_bags(preds)
        gold_bags = self._answer_to_bags(golds)
        f1_per_bag = self._align_bags(pred_bags, gold_bags)
        return np.mean(f1_per_bag)

    def _answer_to_bags(self, answers):
        return [set(self._normalize(answer).split()) for answer in answers]

    def _align_bags(self, pred_bags, gold_bags):
        """ Returns the max metric value over all the answers. """
        scores = np.zeros([len(gold_bags), len(pred_bags)])
        for gold_index, gold_bag in enumerate(gold_bags):
            for pred_index, pred_bag in enumerate(pred_bags):
                if self._is_number_match(pred_bag, gold_bag):
                    scores[gold_index, pred_index] = self._bag_f1(pred_bag, gold_bag)
        row_ind, col_ind = linear_sum_assignment(-scores)
        max_scores = np.zeros([max(len(gold_bags), len(pred_bags))])
        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])
        return max_scores

    def _bag_f1(self, pred_bag, gold_bag):
        intersection = len(gold_bag.intersection(pred_bag))
        if intersection == 0:
            return 0.0
        precision = intersection / float(len(pred_bag)) if pred_bag else 1.0
        recall = intersection / float(len(gold_bag)) if gold_bag else 1.0
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _is_number_match(self, pred_bag, gold_bag):
        pred_numbers = set([word for word in pred_bag if self._is_number(word)])
        gold_numbers = set([word for word in gold_bag if self._is_number(word)])
        if (not gold_numbers) or gold_numbers.intersection(pred_numbers):
            return True
        return False

    def _is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _normalize(self, answer):
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            if not self._is_number(text):
                return "".join(ch for ch in text if ch not in exclude)
            else:
                return text

        def fix_number(text):
            return str(float(text)) if self._is_number(text) else text

        def tokenize(text):
            return re.split(" |-", text)

        tokens = [
            white_space_fix(remove_articles(fix_number(remove_punc(token.lower()))))
            for token in tokenize(answer)
        ]
        tokens = [token for token in tokens if token.strip()]
        normalized = " ".join(tokens).strip()
        return normalized

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "em": mean,
            "f1": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "em": True,
            "f1": True
        }
