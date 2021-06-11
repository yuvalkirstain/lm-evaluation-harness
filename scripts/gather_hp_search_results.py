import json
import os.path
from glob import glob
import pandas as pd


def main():
    tasks = {"mnli", "boolq"}
    for task in tasks:
        data = []
        for file in glob(f"/home/olab/kirstain/lm-evaluation-harness/results/train/{task}/*/results.json"):
            raw_dp = json.load(open(file))
            dp = raw_dp["args"]
            model_args = dp["model_args"]
            for kv in model_args.split(','):
                k, v = kv.split("=")
                dp[k] = os.path.basename(v)
            task = dp["tasks"]
            for k, v in raw_dp[task].items():
                if type(v) == dict:
                    for ik, iv in v.items():
                        dp[ik] = iv
                else:
                    dp[k] = v
            data.append(dp)
        print(f"num experiments = {len(data)}")
        df = pd.DataFrame(data)
        df.to_csv(f"/home/olab/kirstain/lm-evaluation-harness/results/{task}_hp_search_results.csv")


if __name__ == '__main__':
    main()
