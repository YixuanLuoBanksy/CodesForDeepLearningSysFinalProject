"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from ours_models import GPTConfig, GPT_init, GPT_block

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# w/o ddp
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
block_num = 4
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model_block1 = GPT_init(gptconf, block_num)
    model_block2 = GPT_block(gptconf, block_num)
    model_block3 = GPT_block(gptconf, block_num)
    model_block4 = GPT_block(gptconf, block_num)

# crop down the model block size if desired, using model surgery
if block_size < model_block1.config.block_size:
    model_block1.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block2.config.block_size:
    model_block2.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block2.config.block_size:
    model_block3.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block2.config.block_size:
    model_block4.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model_block1.to(device)
model_block2.to(device)
model_block3.to(device)
model_block4.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer_1 = model_block1.configure_optimizers(0.4, learning_rate, (beta1, beta2), device_type)
optimizer_2 = model_block2.configure_optimizers(0.3, learning_rate, (beta1, beta2), device_type)
optimizer_3 = model_block3.configure_optimizers(0.2, learning_rate, (beta1, beta2), device_type)
optimizer_4 = model_block4.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)

checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model_block1 = model_block1
    unoptimized_model_block2 = model_block2
    unoptimized_model_block3 = model_block3
    unoptimized_model_block4 = model_block4
    model_block1 = torch.compile(model_block1) # requires PyTorch 2.0
    model_block2 = torch.compile(model_block2) # requires PyTorch 2.0
    model_block3 = torch.compile(model_block3) # requires PyTorch 2.0
    model_block4 = torch.compile(model_block4) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model_block1.eval()
    model_block2.eval()
    model_block3.eval()
    model_block4.eval()
    for split in ['train', 'val']:
        losses1 = torch.zeros(eval_iters)
        losses2 = torch.zeros(eval_iters)
        losses3 = torch.zeros(eval_iters)
        losses4 = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss1, x_feature = model_block1(X, Y)
                logits, loss2, x_feature = model_block2(x_feature, Y)
                logits, loss3, x_feature = model_block3(x_feature, Y)
                logits, loss4, x_feature = model_block4(x_feature, Y)
            losses1[k] = loss1.item()
            losses2[k] = loss2.item()
            losses3[k] = loss3.item()
            losses4[k] = loss4.item()
        out[split] = [losses1.mean(), losses2.mean(), losses3.mean(), losses4.mean()]
    model_block1.train()
    model_block2.train()
    model_block3.train()
    model_block4.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model_block1 = model_block1 # unwrap DDP container if needed
raw_model_block2 = model_block2 # unwrap DDP container if needed
raw_model_block3 = model_block3 # unwrap DDP container if needed
raw_model_block4 = model_block4 # unwrap DDP container if needed

running_mfu = -1.0

log_file = "4-block-ours-training-log.txt"
with open(log_file, "w", encoding="utf-8") as file:
    file.write("Iteration\tTraining Loss-blk1\tTraining Loss-blk2\tTraining Loss-blk3\tTraining Loss-blk4\tValidation Loss-blk1\tValidation Loss-blk2\tValidation Loss-blk3\tValidation Loss-blk4\n")

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer_1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_2.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_3.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_4.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        losses_train = losses['train']
        losses_val = losses['val']
        
        print(f"step {iter_num}: train loss {losses_train[-1]:.4f}, val loss {losses_val[-1]:.4f}")
        
        
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(f"{iter_num}\t{losses_train[0]:.4f}\t{losses_train[1]:.4f}\t{losses_train[2]:.4f}\t{losses_train[3]:.4f}\t{losses_val[0]:.4f}\t{losses_val[1]:.4f}\t{losses_val[2]:.4f}\t{losses_val[3]:.4f}\n")
        if losses_val[3] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses_val[3]
            if iter_num > 0:
                checkpoint = {
                    'model1': model_block1.state_dict(),
                    'model2': model_block2.state_dict(),
                    'model3': model_block3.state_dict(),
                    'model4': model_block4.state_dict(),
                    'optimizer1': optimizer_1.state_dict(),
                    'optimizer2': optimizer_2.state_dict(),
                    'optimizer3': optimizer_3.state_dict(),
                    'optimizer4': optimizer_4.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt-ours.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss_1, x_feature = model_block1(X, Y)
            logits, loss_2, x_feature = model_block2(x_feature, Y)
            logits, loss_3, x_feature = model_block3(x_feature, Y)
            logits, loss_4, x_feature = model_block4(x_feature, Y)
            loss_1 = loss_1 / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss_2 = loss_2 / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss_3 = loss_3 / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss_4 = loss_4 / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss_1).backward()
        scaler.scale(loss_2).backward()
        scaler.scale(loss_3).backward()
        scaler.scale(loss_4).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer_1)
        scaler.unscale_(optimizer_2)
        scaler.unscale_(optimizer_3)
        scaler.unscale_(optimizer_4)
        torch.nn.utils.clip_grad_norm_(model_block1.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block2.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block3.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block4.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer_1)
    scaler.step(optimizer_2)
    scaler.step(optimizer_3)
    scaler.step(optimizer_4)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer_1.zero_grad(set_to_none=True)
    optimizer_2.zero_grad(set_to_none=True)
    optimizer_3.zero_grad(set_to_none=True)
    optimizer_4.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss_4.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model_block2.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
'''
model_block1.eval()
model_block2.eval()
model_block3.eval()
model_block4.eval()
model_block5 = GPT_block(gptconf, block_num)
model_block6 = GPT_block(gptconf, block_num)
model_block7 = GPT_block(gptconf, block_num)
model_block8 = GPT_block(gptconf, block_num)
model_block9 = GPT_block(gptconf, block_num)
model_block10 = GPT_block(gptconf, block_num)

# crop down the model block size if desired, using model surgery
if block_size < model_block5.config.block_size:
    model_block5.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block6.config.block_size:
    model_block6.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block7.config.block_size:
    model_block7.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block8.config.block_size:
    model_block8.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block9.config.block_size:
    model_block9.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if block_size < model_block10.config.block_size:
    model_block10.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model_block5.to(device)
model_block6.to(device)
model_block7.to(device)
model_block8.to(device)
model_block9.to(device)
model_block10.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer_5 = model_block5.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
optimizer_6 = model_block6.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
optimizer_7 = model_block7.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
optimizer_8 = model_block8.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
optimizer_9 = model_block9.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
optimizer_10 = model_block10.configure_optimizers(0.1, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model_block5 = model_block5
    unoptimized_model_block6 = model_block6
    unoptimized_model_block7 = model_block7
    unoptimized_model_block8 = model_block8
    model_block5 = torch.compile(model_block5) # requires PyTorch 2.0
    model_block6 = torch.compile(model_block6) # requires PyTorch 2.0
    model_block7 = torch.compile(model_block7) # requires PyTorch 2.0
    model_block8 = torch.compile(model_block8) # requires PyTorch 2.0
    model_block9 = torch.compile(model_block9) # requires PyTorch 2.0
    model_block10 = torch.compile(model_block10) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model_block5.eval()
    model_block6.eval()
    model_block7.eval()
    model_block8.eval()
    model_block9.eval()
    model_block10.eval()
    for split in ['train', 'val']:
        losses5 = torch.zeros(eval_iters)
        losses6 = torch.zeros(eval_iters)
        losses7 = torch.zeros(eval_iters)
        losses8 = torch.zeros(eval_iters)
        losses9 = torch.zeros(eval_iters)
        losses10 = torch.zeros(eval_iters)        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss1, x_feature = model_block1(X, Y)
                logits, loss2, x_feature = model_block2(x_feature, Y)
                logits, loss3, x_feature = model_block3(x_feature, Y)
                logits, loss4, x_feature = model_block4(x_feature, Y)
                logits, loss5, x_feature = model_block5(x_feature, Y)
                logits, loss6, x_feature = model_block6(x_feature, Y)
                logits, loss7, x_feature = model_block7(x_feature, Y)
                logits, loss8, x_feature = model_block8(x_feature, Y)
                logits, loss9, x_feature = model_block9(x_feature, Y)
                logits, loss10, x_feature = model_block10(x_feature, Y)                
            losses5[k] = loss5.item()
            losses6[k] = loss6.item()
            losses7[k] = loss7.item()
            losses8[k] = loss8.item()
            losses9[k] = loss9.item()
            losses10[k] = loss10.item()            
        out[split] = [losses5.mean(), losses6.mean(), losses7.mean(), losses8.mean(), losses9.mean(), losses10.mean(),]
    model_block5.train()
    model_block6.train()
    model_block7.train()
    model_block8.train()
    model_block9.train()
    model_block10.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

running_mfu = -1.0
iter_num = 0
log_file = "4-block-ours-training-log.txt"
with open(log_file, "w", encoding="utf-8") as file:
    file.write("Iteration\tTraining Loss-blk1\tTraining Loss-blk2\tTraining Loss-blk3\tTraining Loss-blk4\tValidation Loss-blk1\tValidation Loss-blk2\tValidation Loss-blk3\tValidation Loss-blk4\n")

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer_5.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_6.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_7.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_8.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_9.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_10.param_groups:
        param_group['lr'] = lr
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        losses_train = losses['train']
        losses_val = losses['val']
        
        print(f"step {iter_num}: train loss {losses_train[-1]:.4f}, val loss {losses_val[-1]:.4f}")
        
        

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss_1, x_feature = model_block1(X, Y)
            logits, loss_2, x_feature = model_block2(x_feature, Y)
            logits, loss_3, x_feature = model_block3(x_feature, Y)
            logits, loss_4, x_feature = model_block4(x_feature, Y)
            logits, loss_5, x_feature = model_block5(x_feature, Y)
            logits, loss_6, x_feature = model_block6(x_feature, Y)
            logits, loss_7, x_feature = model_block7(x_feature, Y)
            logits, loss_8, x_feature = model_block8(x_feature, Y)    
            logits, loss_9, x_feature = model_block9(x_feature, Y)
            logits, loss_10, x_feature = model_block10(x_feature, Y) 
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss_5).backward()
        scaler.scale(loss_6).backward()
        scaler.scale(loss_7).backward()
        scaler.scale(loss_8).backward()
        scaler.scale(loss_9).backward()
        scaler.scale(loss_10).backward()        
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer_5)
        scaler.unscale_(optimizer_6)
        scaler.unscale_(optimizer_7)
        scaler.unscale_(optimizer_8)
        scaler.unscale_(optimizer_9)
        scaler.unscale_(optimizer_10)        
        torch.nn.utils.clip_grad_norm_(model_block5.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block6.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block7.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block8.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block9.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model_block10.parameters(), grad_clip)        
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer_5)
    scaler.step(optimizer_6)
    scaler.step(optimizer_7)
    scaler.step(optimizer_8)
    scaler.step(optimizer_9)
    scaler.step(optimizer_10)    
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer_5.zero_grad(set_to_none=True)
    optimizer_6.zero_grad(set_to_none=True)
    optimizer_7.zero_grad(set_to_none=True)
    optimizer_8.zero_grad(set_to_none=True)
    optimizer_9.zero_grad(set_to_none=True)
    optimizer_10.zero_grad(set_to_none=True)
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss_10.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model_block2.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
'''