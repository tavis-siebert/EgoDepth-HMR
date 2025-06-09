## install

```
conda env create -f prohmr.yml
conda activate prohmr
conda activate /work/courses/digital_human/13/envs/prohmr #env installed on cluster
```

## thirdparty submodules
Initialize and update all submodules using 
```
git submodule update --init --recursive
```
If `ModuleNotFoundError`errors occur, you may need to run `pip install -e .` in the relevant third party folder roots

### install nflows

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

## training
To train a model, identify the training script corresponding to the model you wish to train

Example with most important args
```
python train_prohmr_surfnormals_egobody.py 
    --data_root /path/to/data_root \
    --model_cfg /path/to/config \
    --save_dir /path/to/save_dir  \
    --load_pretrained true \
    --checkpoint /path/to/checkpoint
```
Feel free to refer to any of the train.sh scripts as well

## eval (to-do)
To run evaluation, identify the evaluation script corresponding to the model you wish to test

Example
```
python eval_regression_surfnorm_egobody.py --data_root /path/to/data/root --checkpoint /path/to/checkpoint --model_cfg /path/to/config
```

We do not release the test set egocapture_test_smplx.npz.

https://github.com/microsoft/HoloLens2ForCV/blob/main/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py
