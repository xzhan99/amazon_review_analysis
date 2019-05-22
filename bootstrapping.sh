sudo yum -y install git
sudo pip-3.6 install --quiet tensorflow
sudo pip-3.6 install --quiet tensorflow-hub
sudo pip-3.6 install --quiet numpy
sudo pip-3.6 install --quiet jupyter 
sudo pip-3.6 install --quiet scipy 
python3 -c "import nltk;nltk.download('punkt', download_dir='/home/hadoop/nltk_data')" 