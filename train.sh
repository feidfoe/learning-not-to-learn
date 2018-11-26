
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
                  python main.py -e unlearn_0.02  --color_var 0.020 --lr 0.001 --use_pretrain --checkpoint checkpoint/baseline_0.02/checkpoint_step_0100.pth --cuda --is_train --data_split train


