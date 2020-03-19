#! /bin/sh
conda activate torch-xla-nightly
cd
cd "Stereo disparity"
sudo wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && sudo chmod a+x ctpu
export  PATH='/home/dksomisetty/Stereo disparity':$PATH
sudo apt update && sudo apt install -y libsm6 libxext6 libxrender-dev
sudo mkdir -p /mnt/disks/dataset
sudo mount -o discard,defaults /dev/sdb /mnt/disks/dataset
sudo chmod a+w /mnt/disks/dataset
sudo cp /etc/fstab /etc/fstab.backup
echo UUID=`sudo blkid -s UUID -o value /dev/sdb` /mnt/disks/dataset ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
cat /etc/fstab
pip install -r requirements.txt
gcloud auth application-default login

