#https://min.io/docs/minio/linux/index.html

set -ex

wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_20241013133411.0.0_amd64.deb -O minio.deb
sudo dpkg -i minio.deb

sudo systemctl enable minio.service
sudo systemctl status minio.service

rm minio.deb

#https://min.io/docs/minio/linux/reference/minio-mc.html#quickstart
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  -o mc

chmod +x mc
sudo mv mc /usr/local/bin/

mc --help
mc alias set 'local' 'http://192.168.1.174:9000' 'minioadmin' 'minioadmin'

 mc mb local/hub
 mc cp --recursive data/minipile.jsonl local/hub/datasets/minipile-raw/