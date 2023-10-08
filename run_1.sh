# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 1e-4 --ls True --p 0.37  --batch-size 512 --lr 0.2 --cos_anneal True --gates 2 --epoch 240 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 /data/ILSVRC2012 > rn50_gate50lmd10.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --ls True --p 0.37  --batch-size 512 --lr 16e-4 --opt_name ADAMW  --cos_anneal True --gates 2 --epoch 240 --start_epoch_hyper 25 --start_epoch_gl 100 --lmd 10 --grad_mul 5 --reg_w 2.0 /data/ILSVRC2012 > rn50_gate50lmd10_100.txt 2>&1 &

# 0.379 -> p = 0.2958300521469355
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.296 --batch-size 1024 --lr 1e-3 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 0 --grad_mul 5 --reg_w 4.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd0_37_adam_cos_jitter_mixup.txt 2>&1 &
