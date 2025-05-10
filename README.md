## install

```
conda env create -f prohmr.yml
conda activate prohmr
conda activate /work/courses/digital_human/13/envs/prohmr #env installed on cluster
```

## thirdparty submodules

### install nflows

```
cd thirdparty
pip uninstall nflows
pip install -e . nflows/
```


## preprocess

```
python ./preprocess_hha.py input_folder output_folder
```

## training

untar datasets and models first

```
python train_prohmr_egobody_hha_smplx.py --train_dataset_root egobody_release --train_dataset_file egobody_release/smplx_spin_holo_depth_npz/egocapture_train_smplx.npz --val_dataset_root egobody_release --val_dataset_file egobody_release/smplx_spin_holo_depth_npz/egocapture_val_smplx.npz
```


## eval

```
python eval_regression_depth_egobody.py --checkpoint /PATH/TO/MODEL.pt --dataset_root egobody_release
```

We do not release the test set egocapture_test_smplx.npz.

https://github.com/microsoft/HoloLens2ForCV/blob/main/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py
