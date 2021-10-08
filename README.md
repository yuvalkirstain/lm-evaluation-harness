
## A Few More Examples May Be Worth Billions of Parameters

This repo is a fork of the [lm-evaluation-harness repo](https://github.com/EleutherAI/lm-evaluation-harness).
The main changes are that we enable fine-tuning of models and evaluating T5.

### Preliminaries 

Create a virtual env and install requirements:
```bash
conda create -n lm-eval python=3.8
pip install -r requirements.txt
```

Set up your Comet-ml env vars:
```bash
export COMET_API_KEY="<api key>"
```

### Run Experiments (Slurm)

Fill in the `<x>.json` config files in `configs` to make sure that the save paths are ok.

You need to specify two configs - model, and experiments type.

For example, you can run:
```python
python slurm/send_experiments.py --configs slurm/configs/t5/t5_large_basic.json slurm/configs/experiments_fine_tune.json
```

(one can also try first to run:
```python
python slurm/send_experiments.py --configs slurm/configs/t5/t5_large_basic.json slurm/configs/test_slurm_fine_tune.json
```
to make sure that everything works ok)
