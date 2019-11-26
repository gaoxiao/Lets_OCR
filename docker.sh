##拉取基础镜像
docker build -t chineseocr .
##启动服务
docker run -d -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py

docker run --runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=1 \
-d --rm \
-p 8080:8080 \
--shm-size=2g \
--name chineseocr \
chineseocr \
/root/anaconda3/bin/python app.py



docker run --runtime=nvidia \
-it \
-e NVIDIA_VISIBLE_DEVICES=1 \
-v ~/code/chineseocr:/code \
--rm \
-p 8080:8080 \
--shm-size=2g \
--name chineseocr \
chineseocr \
/bin/bash


docker run --runtime=nvidia \
-it \
-e NVIDIA_VISIBLE_DEVICES=1 \
-v ${PWD}:/code \
--rm \
--shm-size=2g \
--name LETS_OCR \
chineseocr \
/bin/bash