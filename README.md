# KimchiVision

Once on El Cap, to check which GPUs are available or in use, type in kernel:

```bash
nvidia-smi
```

To choose a GPU, type:

```bash
export CUDA_VISIBLE_DEVICES=n
```

where n is 0, 1, 2, or 3 depending on the device ID you want to use. After that's set up, if you want to check the GPU you're on, type:

```bash
echo $CUDA_VISIBLE_DEVICES
```

Conda environment setup:
```bash
conda env create -f full_environment.yml -n <env_name>
```
export env 
```bash
conda env export --no-builds > full_environment.yml
```

run notebook in tmux with papermill
```bash
papermill <notebook path> <notebook output path> --log-output

ex: 
papermill our_motion_lstm.ipynb our_motion_lstm_output.ipynb --log-output
```
Run train inparaell
IMPORTANT: MAKE SURE BATCH SIZE IS DIVISIBLE BY NUMBER OF GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 mtr_train.py
```

kill ghost process
```bash
fuser -v /dev/nvidia*
then kill -9 <pid>
```