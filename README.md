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

## training

example

```
python train_prohmr_surfnormals_egobody.py--train_dataset_root egobody_release --train_dataset_file egobody_release/smplx_spin_holo_depth_npz/egocapture_train_smplx.npz --val_dataset_root egobody_release --val_dataset_file egobody_release/smplx_spin_holo_depth_npz/egocapture_val_smplx.npz
```


## eval (to-do)


We do not release the test set egocapture_test_smplx.npz.

https://github.com/microsoft/HoloLens2ForCV/blob/main/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py
