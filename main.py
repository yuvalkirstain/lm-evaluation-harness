import os.path
import comet_ml
import argparse
import json
import numpy as np
import random
import logging

from lm_eval import models, tasks, evaluator, base
from lm_eval.utils import simple_parse_args_string

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    parser.add_argument('--train_args', default="")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_path):
        print(f"Output path {args.output_path} exists!!!")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    lm = models.get_model(args.model)

    train_args = simple_parse_args_string(args.train_args)
    model_args = simple_parse_args_string(args.model_args)

    if train_args:
        train_args.update(model_args)
        train_args["seed"] = args.seed

    results = evaluator.evaluate(lm, task_dict, args.provide_description, args.num_fewshot, args.limit, train_args,
                                 args.model_args, args.seed)

    results["args"] = args.__dict__
    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    for task, task_res in results.items():
        if task not in task_names:
            continue
        if "train_args" not in task_res:
            experiment = comet_ml.Experiment(
                api_key=os.environ.get('COMET_API_KEY'),
                project_name=os.environ.get('COMET_PROJECT', "few-shot"),
                workspace=os.environ.get('COMET_WORKSPACE', "yuvalkirstain"),
            )
            experiment.log_asset(args.output_path)
        else:
            experiment = comet_ml.ExistingExperiment(api_key=os.environ.get('COMET_API_KEY'),
                                                     previous_experiment=task_res["train_args"]["previous_experiment"])
            experiment.log_asset(args.output_path)


if __name__ == "__main__":
    main()
