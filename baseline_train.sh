

gpuid=0
datapath=<path_to_data>
checkpointpath=<path_to_checkpoint>

nvidia-docker run -v `pwd`:`pwd` -w `pwd` --rm \
                  -v ${datapath}:`pwd`/dataset/ \
                  -v ${checkpointpath}:`pwd`/checkpoint/ \
                  --name 'mnist_color_baseline_gpu'${gpuid} \
                  -e CUDA_VISIBLE_DEVICES=${gpuid} \
                  -it \
                  --ipc=host \
                  feidfoe/pytorch:latest \
                  python main.py -e baseline_0.02  --color_var 0.020 --lr 0.01 --cuda --train_baseline --data_split train


