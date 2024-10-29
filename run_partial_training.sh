# sleep 60
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init bash
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init zsh
# source ~/.bashrc
# source ~/.zshrc
# conda activate rd-pruning
pip install --upgrade pip
pip install -r requirements.txt
python partial_training_image.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --num-clients ${5} --client-select-ratio ${6} --local-train-ratio ${7} --rounds ${8} --epochs ${9} --batch-size ${10} --lr ${11} --weight-decay ${12} --eta_g ${13} --dynamic-eta_g ${14} --norm-type ${15} --client-capacity-distribution ${16} --global-lr-decay ${17} --gamma ${18} --data-augmentation ${19} >> ${20}