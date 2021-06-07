import json
import os
import subprocess
from pathlib import Path
import argparse


def create_experiment_dir(results_dir, train, task, model_base_name, n_shot, dropout, decay, lr, optimizer, gradient_clip_val, min_step):
    exp_dir = os.path.join(results_dir,
                           train,
                           task,
                           f"{model_base_name}-{task}-{n_shot}-{dropout}-{decay}-{lr}-{optimizer}-{gradient_clip_val}-{min_step}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    return exp_dir


def create_slurm_scripts(model_base_name, cur_output_dir, project_name, model, task, n_shot, cur_output_path, dropout,
                         decay, lr, train, optimizer, model_type, lr_scheduler_type, train_batch_size, grad_accu,
                         save_pre, gradient_clip_val, gpus, min_step):
    script_name = os.path.join(cur_output_dir, "run.sh")
    slurm_name = os.path.join(cur_output_dir, "slurm.sh")
    with open(slurm_name, "w") as f:
        script = f"""#!/bin/bash -x
#SBATCH --job-name=few-shot-{model_base_name}-hp-tune
#SBATCH --output={os.path.join(cur_output_dir, "slurm.out")}
#SBATCH --error={os.path.join(cur_output_dir, "slurm.err")}
#SBATCH --time=1440
#SBATCH --signal=USR1@120
#SBATCH --partition="killable"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000 
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n-101,n-007
#SBATCH --gpus={gpus}
srun sh {script_name}
"""
        f.write(script)

    with open(script_name, "w") as f:
        run_script = f"""#!/bin/bash -x
cd /home/olab/kirstain/lm-evaluation-harness
export COMET_PROJECT="{model_base_name}-{project_name}"
bash slurm/run_single_experiment.sh {model} {task} {n_shot} {cur_output_path} {dropout} {decay} {lr} {train} {optimizer} {model_type} {lr_scheduler_type} {train_batch_size} {grad_accu} {save_pre} {gradient_clip_val} {min_step}
"""
        f.write(run_script)

    subprocess.Popen(["chmod", "ug+rx", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return slurm_name


def send_job_and_report(slurm_name):
    process = subprocess.Popen(["sbatch", slurm_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("output:")
    print(stdout.decode("utf-8"))
    print("err:")
    print(stderr.decode("utf-8"))


def run_model_jobs(models, tasks, n_shots, results_dir, project_name, model_type, gpus, dropouts=("no",),
                   decays=("no",), lrs=("no",), optimizers=("no",),
                   lr_scheduler_type="no", batch_and_grad_accu=("no",), temp_dir="no", gradient_clip_vals=("no",),
                   min_steps=("no,")):
    sent_jobs = 0
    for model in models:
        for task in tasks:
            for n_shot in n_shots:
                for dropout in dropouts:
                    for decay in decays:
                        for optimizer in optimizers:
                            for gradient_clip_val in gradient_clip_vals:
                                for min_step in min_steps:
                                    for lr in lrs:
                                        for train_batch_size, grad_accu in batch_and_grad_accu:
                                            if n_shot == 0:
                                                train = "no_train"
                                                n_shot = 32
                                            else:
                                                train = "train"
                                            model_base_name = os.path.basename(model)
                                            cur_output_dir = create_experiment_dir(results_dir, train, task,
                                                                                   model_base_name,
                                                                                   n_shot, dropout, decay, lr,
                                                                                   optimizer, gradient_clip_val,
                                                                                   min_step)

                                            cur_output_path = os.path.join(cur_output_dir, "results.json")
                                            if os.path.exists(cur_output_path):
                                                print(f"{cur_output_path} exists")
                                                continue
                                            slurm_name = create_slurm_scripts(model_base_name, cur_output_dir,
                                                                              project_name,
                                                                              model, task, n_shot,
                                                                              cur_output_path, dropout, decay, lr,
                                                                              train,
                                                                              optimizer, model_type,
                                                                              lr_scheduler_type, train_batch_size,
                                                                              grad_accu,
                                                                              temp_dir, gradient_clip_val, gpus, min_step)
                                            send_job_and_report(slurm_name)
                                            print(slurm_name)
                                            sent_jobs += 1

    print(f"Sent {sent_jobs} jobs")


def main():
    parser = argparse.ArgumentParser(description='Configuration',
                                     allow_abbrev=False)
    parser.add_argument('--configs', type=str, required=True, nargs="+")
    args = parser.parse_args()
    assert len(args.configs) == 2, "You need to have an experiment and model configs"
    run_args = {}
    for config in args.configs:
        cur_conf = json.load(open(config))
        assert len(run_args.keys() & cur_conf.keys()) == 0, "one of the configs overrides the other."
        run_args.update(cur_conf)
    run_model_jobs(**run_args)


if __name__ == '__main__':
    main()
