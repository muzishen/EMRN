CUDA_VISIBLE_DEVICES=0  python train_double.py  \
--warm_epoch 10  \
--stride 1  \
--data_dir  ./pytorch  \
--batchsize 32  \
--erasing_p 0.5 \
--name 20190927_double_128_256 \
--train_all \
--resume





