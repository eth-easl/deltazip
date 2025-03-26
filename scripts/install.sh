mount /dev/vdb /mnt
apt update && apt upgrade
apt install -y python3-pip
pip install transformers tabulate matplotlib seaborn