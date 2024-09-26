sudo apt-get update -y && sudo apt-get install -y openssh-server

cd ~
wget https://ocs-cn-north1.oppoer.me/mlp-resource/remote-dev/ssh.zip
unzip ssh.zip
sudo cp -rf ssh/ssh_host* /etc/ssh/
sudo chmod 0600 /etc/ssh/*key
sudo mkdir -p /run/sshd

sshd_conf=/etc/ssh/sshd_config
sudo sed -r -i 's/^(#UseDNS yes|UseDNS yes)/UseDNS no/' $sshd_conf
sudo sed -r -i 's/^(#Port 22|Port 22)/Port 18822/' $sshd_conf
sudo sed -i 's/GSSAPIAuthentication yes/GSSAPIAuthentication no/g' $sshd_conf
sudo sed -r -i 's/#(ListenAddress 0.0.0.0)/\1/g' /etc/ssh/sshd_config
sudo sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 60/' $sshd_conf
sudo sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 0/' $sshd_conf
sudo sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords no/' $sshd_conf
sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/g' $sshd_conf
sudo sed -i 's/#HostbasedAuthentication no/HostbasedAuthentication no/' $sshd_conf
sudo sed -i 's/#RhostsRSAAuthentication no/RhostsRSAAuthentication no/' $sshd_conf
sudo sed -i 's/X11Forwarding yes/X11Forwarding no/' $sshd_conf
sudo sed -i 's/GSSAPIAuthentication yes/GSSAPIAuthentication no/' $sshd_conf
sudo sed -i 's/#IgnoreRhosts no/IgnoreRhosts yes/' $sshd_conf
sudo sed -i 's/#*PermitRootLogin .*/PermitRootLogin yes/' $sshd_conf
sudo sed -i 's/#*PubkeyAuthentication .*/PubkeyAuthentication yes/' $sshd_conf
sudo chmod 600 $sshd_conf
sudo chown root:root $sshd_conf

mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

nohup sudo /usr/sbin/sshd -D > /dev/null 2>&1 &

echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC28JuEef4EJRwTqbmi6ScHobtnIVV3ZrCu77B/MjPVMWwmwN3dcVEqHRa1x6FE6urvKR3xikxUeKDxF62p2atznRKEL0+1VPJjTch+sDsc9BbrJP/WYD1NRmPifIlj6nCIH1//RbUcu5P0G7HvpTx3DklWyM2tlTOnrk+CnUrqhQQP4MsjhzgLUf2G3N2VBw20Yz6jCkuSeyKCNni6A1LsbV145qc67dxzg9xR3f4Xur4Y0szfj10knn5MEUX1jXXKTGUrOHhNswzK1Vh46MUCmhnooLCxJy4QUr5ABueG4OvCHDOalLfC30MhkVl1wXAQaU3ogR8GRe2lqzDMTElwpHga4JoHv62+nuWOrml07Qhoqyoq5FVzMTf3SPG9NC7kr7kQFKuOHnnklLVkS/06UgaoftOff7uCVbhZV7mCjfIBfViq7mwB61Ka5yTvaTAYoEtJO8Jj+C5YhRbwGivOEpgosVCfr2AzIVKLA/4RTho2xZFR9sbDi8N8+7J02J0= adc\s9056161@NS9056161" >> ~/.ssh/authorized_keys

sudo apt install zsh -y

/home/notebook/code/personal/S9056161/miniconda3/bin/conda init bash
/home/notebook/code/personal/S9056161/miniconda3/bin/conda init zsh