sudo yum update

sudo yum install -y \
    apache2 \
    apache2-dev \
    libapache2-mod-wsgi-py3 \
    python3 \
    python3-pip

if [ -d /home/heartlab-api ]; then
  sudo rm -R /home/heartlab-api
  mkdir /home/heartlab-api
fi
