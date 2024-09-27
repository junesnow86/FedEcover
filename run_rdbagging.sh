sleep 60
sh /home/notebook/code/personal/S9056161/init.sh
source ~/.bashrc
conda activate rd
python partial_training_image.py --method rdbagging --model resnet --distribution non-iid --alpha 0.5 --epochs 20 --save-dir results/0919 >> logs/0919/rdbagging_resnet_cifar10_alpha0.5_epochs20.log