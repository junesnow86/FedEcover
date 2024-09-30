# sleep 60
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init bash
# /home/notebook/code/personal/S9056161/miniconda3/bin/conda init zsh
# source ~/.bashrc
# source ~/.zshrc
# conda activate rd-pruning
pip install -r requirements.txt
python homo_training_image.py --method $1 --model $2 --dataset $3 --distribution $4 --alpha $5 --rounds $6 --epochs $7 --batch-size $8 >> $9