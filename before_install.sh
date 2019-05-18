sudo yum update

source activate tensorflow_p36

sudo yum install -y \
    apache2 \
    apache2-dev \
    libapache2-mod-wsgi-py3

if [ -d /home/heartlab-api ]; then
  sudo rm -R /home/heartlab-api
  mkdir /home/heartlab-api
fi
