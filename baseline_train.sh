

gpuid=0

nvidia-docker run -v `pwd`:`pwd` -w `pwd` --rm \
                  -v ~/bj/unk_unks/unlearn/mnist/dataset/:`pwd`/dataset/ \
                  -v ~/bj/unk_unks/unlearn/mnist/final/checkpoint/:`pwd`/checkpoint/ \
                  --name 'mnist_color_baseline_gpu'${gpuid} \
                  -e CUDA_VISIBLE_DEVICES=${gpuid} \
                  -it \
                  --ipc=host \
                  feidfoe/pytorch:latest \
                  python main.py -e baseline_0.02  --color_var 0.020 --lr 0.01 --cuda --train_baseline --data_split train


