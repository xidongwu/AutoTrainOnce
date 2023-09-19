# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage baseline --weight-decay 5e-2 --lr 8e-4 --cos_anneal True --ls True  --opt_name ADAMW --epoch 125  /data/ILSVRC2012 > rn50_bl.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=4,5 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --p 0.37  --batch-size 256 --lr 8e-4 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 /p/federatedlearning/data/ILSVRC2012 > rn50_lmd10_50_adam_singlecosine.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.308 --batch-size 256 --lr 8e-4 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd10_40_adam_1cos_jitter_mixup.txt 2>&1 &
