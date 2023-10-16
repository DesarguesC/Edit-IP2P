# Edit Anything


## Train Projection Model

### environment variables

```bash
export RANK=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2 # (your gpus rank)
export BSIZE=32
export NUM_EPOCH=10000
export IMAGE_FOLDER="../autodl-tmp/..." # (your dataset path)
```

install nccl dependencies
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <YOUR GPU AMOUNT> # test
```


train
```bash
python -m torch.distributed.launch --nproc_per_node=3 --master_port=1010 train_seg2latent.py
```


### Train out Control Net with pretrained projection model

There's only need to set environment variables for multi GPU this time.

```bash
export RANK=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2
```

Then launch python process via torch.distributed.launch
```bash
python -m torch.distributed.launch --nproc_per_node=3 --master_port=1010 train_seg_control.py --gpus 0,1,2 --image_folder ../autodl-tmp/DATASET \
       --sd_ckpt ./checkpoints/v1-5-pruned-emaonly.ckpt
```














