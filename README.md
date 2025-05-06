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


## training

untar datasets and models first

```
python train_prohmr_depth_egobody.py --data_source real --train_dataset_root egobody_release --val_dataset_root egobody_release
```


## eval

```
python eval_regression_depth_egobody.py --checkpoint /PATH/TO/MODEL.pt --dataset_root egobody_release
```

We do not release the test set egocapture_test_smplx.npz.

https://github.com/microsoft/HoloLens2ForCV/blob/main/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py
