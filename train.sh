
gpuid=0


nvidia-docker run -v `pwd`:`pwd` -w `pwd` --rm \
                  --name 'mnist_color_baseline_gpu'${gpuid} \
                  -e CUDA_VISIBLE_DEVICES=${gpuid} \
                  -it \
                  --ipc=host \
                  feidfoe/pytorch:latest \
                  python main.py -e unlearn_0.02  --color_var 0.020 --lr 0.001 --use_pretrain --checkpoint path/to/pretrained/model.pth --cuda --is_train --data_split train


