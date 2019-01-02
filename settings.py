import os
import logging

num_classes =345
iter_sche = [70000, 80000, 100000, 150000]
attention_type = 'use_default_input_feature' # 'use_l2_normalized_feature' or 'use_default_input_feature'
use_toy = False
delf_init = False
classification_model = 'delf'  # res or delf

aug_data = False

batch_size = 64
lr = 5e-4

data_dir = '/data/yehr/quick_draw'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 10000
latest_steps = 100
val_step = 10

num_workers = 60
num_gpu = 2
device_id = '2,3'


# for showing logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


