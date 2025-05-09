## install

```
conda env create -f prohmr.yml
conda activate prohmr
conda activate /work/courses/digital_human/13/envs/prohmr #env installed on cluster
```

## reinstall nflows

```
pip uninstall nflows
git clone git@github.com:nkolot/nflows.git
cd nflows
pip install -e .
```

## install Dep2HHA

```
git clone git@github.com:charlesCXK/Depth2HHA-python.git
mv Depth2HHA prohmr/utils
```

## preprocess

```
python ./prohmr/utils/preprocess_hha.py --input_folder --output_folder
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
