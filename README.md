## install
Create and activate a virtual environment using your preferred method
Example using Python venv
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
Install requirements in `requirements.txt`
```
pip install -r requirements.txt
```

## thirdparty submodules
Initialize and update all submodules using 
```
git submodule update --init --recursive
```
If `ModuleNotFoundError`errors occur, you may need to run `pip install -e .` in the relevant third party folder roots

### install nflows
NOTE: this step might not be needed if your pip install worked properly. To see, try skipping this step first and come back if needed.
```
cd thirdparty
pip uninstall nflows
pip install -e . nflows/
```

## setup data dir
For our project, all relevant data was moved to a folder on the cluster.
To ensure compatability with our setup, ensure you have a `data_root` (name it whatever you wish of course) folder with the following structure
```
data_root/
├── egobody_release/
│   ├── egocentric_depth_processed/
│   ├── smplx_spin_holo_depth_npz/
├── data/
│   ├── datasets/
│   ├── smpl/
│   ├── smplx_model/
│   ├── smplx_to_smpl.npz
│   └── smpl_mean_params.npz
```
## config
Refer to the README and configs in `prohmr/configs`

## training
To train a model, identify the training script corresponding to the model you wish to train

Example with most important args
```
python train_prohmr_surfnormals_egobody.py \
    --data_root /path/to/data_root \
    --model_cfg /path/to/config \
    --save_dir /path/to/save_dir  \
    --load_pretrained true \
    --checkpoint /path/to/checkpoint
```
Feel free to refer to any of the train.sh scripts as well

## eval
To run evaluation, identify the evaluation script corresponding to the model you wish to test

Example
```
python eval_regression_surfnorm_egobody.py --data_root /path/to/data/root --checkpoint /path/to/checkpoint --model_cfg /path/to/config
```

IMPORTANT: make sure the config matches the one you used to train the model, otherwise there might be undefined behavior (this mostly applies to having the matching MODEL.FLOW.MODE in the config)

## Reproduce the result
Our best model is the fusion model with `concat` strategy.

IMPORTANT: You might need to modify `SMPL.MODEL_PATH` and `SMPL.MEAN_PARAMS` in the config files to your own path to the smpl data.

### Surface normal
- Run:
```
python eval_regression_surfnorm_egobody.py --data_root /path/to/data/root --checkpoint /path/to/checkpoint --model_cfg prohmr/configs/prohmr.yaml
```
- Script: `eval_regression_surfnorm_egobody.py`
- Config: `prohmr/configs/prohmr.yaml`

### All fusion models
We set `MODEL.FLOW.MODE` in the config to either `concat`, `attention`, or `mlp` according to the fusion strategy.
- Run:
```
python eval_regression_fusion _egobody.py --data_root /path/to/data/root --checkpoint /path/to/checkpoint --model_cfg prohmr/configs/prohmr_fusion.yaml
```
- Script: `eval_regression_fusion_egobody.py`
- Config:  `prohmr/configs/prohmr_fusion.yaml`
