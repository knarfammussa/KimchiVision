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
conda create --name <env> --file requirements.txt
```