# sleep 60
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init bash
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init zsh
# source ~/.bashrc
# source ~/.zshrc
# conda activate rd-pruning
# taskset -c 0-4 python partial_training_image.py --method fedrd --model cnn --dataset cifar10 --distribution alpha0.5 --num-clients 10 --client-select-ratio 1.0 --local-train-ratio 1.0 --rounds 300 --epochs 2 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --eta_g 1.0 --dynamic-eta_g False --norm-type sbn --client-capacity-distribution 2 --global-lr-decay True --gamma 0.9 --data-augmentation True >> logs/1108/fedrd-cnn-cifar10-alpha0.5-capacity2-10clients.log
# taskset -c 20-23 python partial_training_image.py --method fedrame --model resnet --dataset tiny-imagenet --distribution alpha0.5 --num-clients 10 --client-select-ratio 1.0 --local-train-ratio 1.0 --rounds 300 --epochs 10 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --eta_g 1.0 --dynamic-eta_g False --norm-type sbn --client-capacity-distribution 2 --global-lr-decay False --gamma 1.0 --data-augmentation True --num-workers 4 >> logs/1108/fedrame2-no-gsd-resnet-tinyimagenet0-alpha0.5-capacity2-10clients.log
pip install --upgrade pip
pip install -r requirements.txt
python partial_training_image.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --num-clients ${5} --client-select-ratio ${6} --local-train-ratio ${7} --rounds ${8} --epochs ${9} --batch-size ${10} --lr ${11} --weight-decay ${12} --eta_g ${13} --dynamic-eta_g ${14} --norm-type ${15} --client-capacity-distribution ${16} --global-lr-decay ${17} --gamma ${18} --data-augmentation ${19} >> ${20}