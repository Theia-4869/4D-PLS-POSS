import numpy as np
import yaml
from operator import itemgetter
import struct

np.set_printoptions(threshold=np.inf)

config_file = "data/SemanticPoss/semantic-poss.yaml"
with open(config_file, 'r') as stream:
    doc = yaml.safe_load(stream)
    all_labels = doc['labels']
    learning_map_inv = doc['learning_map_inv']

label_file = "data/SemanticPoss/sequences/02/labels/000001.label"
prediction_file = "test/pretrain_true_frame_1_epoch_400_bsz_32_lr_1e-2_importance_None_str1_bigpug_1/probs/02_0000000_i.npy"

# frame_labels = np.fromfile(label_file, dtype=np.uint32)
# sem_labels = frame_labels & 0xFFFF
# ins_labels = frame_labels >> 16
# print(sem_labels.shape)
# print(sem_labels[sem_labels!=0][:30])
# print(ins_labels[ins_labels!=0])

pred_labels = np.load(prediction_file)
sem_labels = pred_labels & 0xFFFF
sem_labels = itemgetter(*sem_labels)(learning_map_inv)
sem_labels = np.array(sem_labels, dtype=np.uint32)
ins_labels = pred_labels >> 16
# print(sem_labels.shape)
# print(sem_labels)
# print(ins_labels[ins_labels!=0][:30])