import os
import numpy as np
import yaml
from operator import itemgetter
import struct

config_file = "data/SemanticPoss/semantic-poss.yaml"
with open(config_file, 'r') as stream:
    doc = yaml.safe_load(stream)
    all_labels = doc['labels']
    learning_map_inv = doc['learning_map_inv']

label_dir = "data/SemanticPoss/sequences/02/labels"
# prediction_dir = "test/pretrain_true_frame_4_epoch_400_bsz_32_lr_1e-2_importance_None_str1_bigpug_4/probs"
prediction_dir = "test/pretrain_false_frame_4_epoch_300_bsz_16_lr_1e-3_importance_None_str1_bigpug_4/probs"
output_dir = prediction_dir.replace("probs", "outputs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(500):
    print('{}/500'.format(i))
    label_file = '{}/{:06d}.label'.format(label_dir, i+1)
    frame_labels = np.fromfile(label_file, dtype=np.uint32)

    semantic_file = '{}/02_{:07d}.npy'.format(prediction_dir, i)
    semantic_labels = np.load(semantic_file)
    instance_file = '{}/02_{:07d}_i.npy'.format(prediction_dir, i)
    instance_labels = np.load(instance_file)

    if frame_labels.shape[0] != semantic_labels.shape[0]:
        raise ValueError('number of points {} in .label must be equal to number of points {} in .npy'.format(frame_labels.shape[0], semantic_labels.shape[0]))
    if semantic_labels.shape[0] != instance_labels.shape[0]:
        raise ValueError('number of points {} in .npy must be equal to number of points {} in _i.npy'.format(semantic_labels.shape[0], instance_labels.shape[0]))

    sem_labels = semantic_labels & 0xFFFF
    sem_labels = itemgetter(*sem_labels)(learning_map_inv)
    sem_labels = np.array(sem_labels, dtype=np.uint32)
    ins_labels = np.array(instance_labels, dtype=np.uint32)
    pred_labels = sem_labels + (ins_labels << 16)
    output_file = '{}/{:06d}.label'.format(output_dir, i+1)
    with open(output_file, 'wb')as fp:
        for label in pred_labels:
            l = struct.pack('I', label)
            fp.write(l)