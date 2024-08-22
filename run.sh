sleep 60
sh /home/notebook/code/personal/S9056161/init.sh
source ~/.bashrc
conda activate rd
python rd_base.py >> rd_base.log
