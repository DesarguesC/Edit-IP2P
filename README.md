# Edit Anything


## Train Projection Model

### environment variables

```bash
export RANK=0
export WORLD_SIZE=3     # the number of the thread to be called
export MASTER_ADDR=localhost
export MASTER_PORT=5678
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=... # (your gpus rank)

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










