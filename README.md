## Install

```
conda env create -f prohmr.yml
conda activate prohmr
```

## Training

untar datasets and models first

```
python train_prohmr_depth_egobody.py --data_source real --train_dataset_root egobody_release --val_dataset_root egobody_release
```


## Eval

```
python eval_regression_depth_egobody.py --checkpoint /PATH/TO/MODEL.pt --dataset_root egobody_release
```

We do not release the test set egocapture_test_smplx.npz.

https://github.com/microsoft/HoloLens2ForCV/blob/main/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py

## Branch-specific \[VPoser\]

To setup VPoser
1. Clone [human_body_prior repo](https://github.com/nghorbani/human_body_prior) in the project root
2. Follow the instructions on the repo for setup (i.e. `cd` into the root folder of the `human_body_prior` repo; install requirements; run `setup.py`)\
&nbsp;- There is an `__init__.py` file in the root which might require deletion if setup doesn't properly resolve module loading errors
3. Sign up for access on the [SMPL-X website](https://smpl-x.is.tue.mpg.de/index.html)
4. Once you have access, go to Downloads, and download VPoser 2.0
5. Place the model folder in the root of the `human_body_prior` repo (or wherever you think works, just be sure to change the path accordingly in your config). The downloaded model folder should be named something like `V02_05` and contain a snapshots folder with `.ckpt` files containing the models

NOTE: in order to train your model, you must first change the `config.model.vposer.expr_dir` argument in `prohmr/configs/prohmr.yaml` to the correct path to your downloaded VPoser.
Then, you can run the scripts as above. I've tweaked the training and evaluation slightly to be more compute-friendly and run the new `HMRDepthEgoBodyVPoser` off-the-shelf. 

If I missed anything (e.g. extra pip installs), I'm sorry. I know I didn't setup perfectly myself, but it all worked out in the end handling errors as they came. There shouldn't be any awful version conflicts or anything like that.