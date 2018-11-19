
nvidia-docker run -v `pwd`:`pwd` --rm \
                  -v ~/bj/unk_unks/unlearn/mnist/dataset/:`pwd`/dataset/ \
                  -v ~/bj/unk_unks/unlearn/mnist/final/hybrid/pretrained/:`pwd`/pretrained/ \
                  --name 'mnist_color_hybrid_gpu2_' \
                  -w `pwd` \
                  -e CUDA_VISIBLE_DEVICES=2 \
                  -it \
                  --ipc=host \
                  feidfoe/pytorch:latest \
                  python main.py -e test_  --color_var 0.020 --lr 0.001 --use_pretrain --cuda --is_train --data_split train


