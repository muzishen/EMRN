CUDA_VISIBLE_DEVICES=2  python train_triplet.py  \
--warm_epoch 10  \
--stride 1  \
--data_dir  ./pytorch  \
--batchsize 32  \
--erasing_p 0.5 \
--name 20190927_triplet \
--train_all \
--resume




