# sleep 60
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init bash
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init zsh
# source ~/.bashrc
# source ~/.zshrc
# conda activate rd-pruning
pip install -r requirements.txt
python partial_training_image.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --rounds ${5} --epochs ${6} --batch-size ${7} --eta_g ${8} --dynamic-eta_g ${9} >> ${10}