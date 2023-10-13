# Edit Anything


## Train Projection Model

### environment variables

```bash
export RANK=0
export CUDA_VISIBLE_DEVICES=... # (your gpus rank)
export IMAGE_FOLDER=... # (your dataset path)
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










