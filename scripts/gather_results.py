import json
import os.path
from glob import glob
import pandas as pd


def main():
    data = []
    tasks = set()
    results_path = "/home/olab/kirstain/lm-evaluation-harness/results_arzi/results_combined.csv"
    for file in glob("/home/olab/kirstain/lm-evaluation-harness/results_arzi/*/*/*/results.json"):
        try:
            raw_dp = json.load(open(file))
        except:
            print(f"check this {file}")
            continue
        dp = raw_dp["args"]
        model_args = dp["model_args"]
        if dp["train_args"] == "":
            dp["train"] = False
        else:
            dp["train"] = True
        for kv in model_args.split(','):
            k, v = kv.split("=")
            dp[k] = os.path.basename(v)
        task = dp["tasks"]
        tasks.add(task)
        for k, v in raw_dp[task].items():
            if type(v) == dict:
                for ik, iv in v.items():
                    dp[ik] = iv
            else:
                dp[k] = v
        data.append(dp)
    print(f"num of tasks = {len(tasks)} ; num experiments = {len(data)}")
    df = pd.DataFrame(data)
    # old_df = pd.read_csv(results_path, index_col=False)
    # df = pd.concat([df, old_df])
    cols = [c for c in df.columns if "unnamed" not in c.lower()]
    df = df[cols]
    print(len(df))
    df = df.drop_duplicates(ignore_index=True)
    print(len(df))       
    df.to_csv(results_path)


if __name__ == '__main__':
    main()
