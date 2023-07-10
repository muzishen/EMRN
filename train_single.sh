CUDA_VISIBLE_DEVICES=1  python train_single.py  \
--warm_epoch 10  \
--stride 1  \
--data_dir  ./pytorch  \
--batchsize 32  \
--erasing_p 0.5 \
--name 20190927_single \
--train_all \
--resume





