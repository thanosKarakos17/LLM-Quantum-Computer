# config.py - a minimal config file for training QAOA-GPT with FEATHER

out_dir = 'out-qaoa-maxcut'
data_dir = 'data/maxcut'  # where train.bin/test.bin and *.npy live

# model
n_layer = 6
n_head = 10
n_embd = 100  # this must match feather_dim
block_size = 103  # max length of tokenized input
bias = False

# training
batch_size = 32
learning_rate = 3e-4
max_iters = 20000
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# optimizer
weight_decay = 1e-1

# device
device = 'cuda'  # or 'cpu'
dtype = 'bfloat16'
compile = False

# feather
feather_dim = 100  # this must match n_embd
