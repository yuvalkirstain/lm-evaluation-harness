
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

Fill in the `hp_tune.json` to make sure that the save are ok.

run:
```python
python slurm/send_experiments.py --configs slurm/configs/t5/t5_large_basic.json slurm/configs/experiments_fine_tune.json
```

(one can also try first to run:
```python
python slurm/send_experiments.py --configs slurm/configs/t5/t5_large_basic.json slurm/configs/test_slurm_fine_tune.json
```
so make sure things are ok)