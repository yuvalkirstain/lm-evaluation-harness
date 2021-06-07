
## Data VS Params

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

### Run hp-tuning experiments

Fill in the:
1. `save.json` file to 

```python
python slurm/send_experiments.py --configs slurm/configs/save.json slurm/configs/t5/t5_large_basic.json slurm/configs/train.json slurm/configs/hp_tune.json
```