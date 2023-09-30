# PESF-KD
PyTorch code for our TMM 2023 paper "Parameter-Efficient and Student-Friendly Knowledge Distillation".
# Usage
## Folder cv contains the Computer Vision project (CIFAR-100 & ImageNet).

Use scripts(cv/scripts) or command     
``` shell
    python train_student_DDP.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1     
```
to start the training.
    
## Folder nlp contains the Natural Language Processing project (GLUE).

Use the following command to fine-tune the teacher model.
``` shell
python run_glue.py
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name QQP \
--do_train \
--do_eval \
--do_lower_case \
--data_dir glue_data/QQP \
--max_seq_length 128 \
--per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--learning_rate 2e-5 \
--num_train_epochs 4.0 \
--output_dir /tmp/QQP/
``` 
Use the following command to train the student model.
``` shell
python run_glue_distillation.py \
--model_type bert \
--teacher_model /tmp/QQP \
--student_model bert-base-uncased \
--task_name QQP \
--num_hidden_layers 6 \
--alpha 0.7 \
--beta 100.0 \
--do_train \
--do_eval \
--do_lower_case \
--data_dir ./glue_data/QQP \
--max_seq_length 128 \
--per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--learning_rate 5e-5 \
--num_train_epochs 4 \
--output_dir ./tmp/QQP
```

# Acknowledgments
The implementation of image classification is based on https://github.com/HobbitLong/RepDistiller

The implementation of text classification is based on https://github.com/bzantium/pytorch-PKD-for-BERT-compression

# Citation
```
@article{pesf-kd,
author    = {Jun Rao and Xv Meng and Liang Ding and Shuhan Qi and Xuebo Liu and Min Zhang and Dacheng Tao},
  title     = {Parameter-Efficient and Student-Friendly Knowledge Distillation},
  journal   = {{IEEE} Trans. Multim.},
  pages     = {1--12},
  year      = {2023},
  url       = {},
  doi       = {10.1109/TMM.2023.3321480}

```
