
docker run --runtime=nvidia \
-it \
-e NVIDIA_VISIBLE_DEVICES=1 \
-v ${PWD}:/code \
--rm \
--shm-size=2g \
--name LETS_OCR \
lets_ocr \
/bin/bash