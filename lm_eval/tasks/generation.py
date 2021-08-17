import datasets

from lm_eval import metrics
from lm_eval.base import rf
from lm_eval.tasks.common import HFTask


class XSum(HFTask):
    DATASET_PATH = "gem"
    DATASET_NAME = "xsum"

    def __init__(self):
        self.data = None
        super().__init__()

    def download(self):
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)

    def doc_to_text(self, doc):
        return f'{doc["document"]}\nShort summary:'

    def doc_to_target(self, doc):
        target = doc["target"] if "target" in doc else doc["references"][0]
        return " " + target

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
        return rf.greedy_until(ctx, ["\n"])

    def process_results(self, doc, results):
        references = doc['references']
        prediction = results[0]
        return metrics.rouge(references, prediction)

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "rouge1": metrics.mean,
            "rouge2": metrics.mean,
            "rougeL": metrics.mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
        }


class CommonGen(XSum):
    DATASET_NAME = "common_gen"

    def doc_to_text(self, doc):
        return f'Write a short sentence with the words {", ".join(doc["concepts"][:-1])}, and {doc["concepts"][-1]}:'

    def has_test_docs(self):
        """Whether the task has a test set"""
        return False


class Asset(XSum):
    DATASET_PATH = "asset"
    DATASET_NAME = "simplification"

    def download(self):
        super().download()
        self.data["train"] = self.data["validation"]
        self.data["validation"] = self.data["test"]
        self.data = self.data.rename_column("original", "document")
        self.data = self.data.rename_column("simplifications", "references")

    def doc_to_text(self, doc):
        return f'Original: {doc["document"]}\nSimplified:'


## TODO style transfer!