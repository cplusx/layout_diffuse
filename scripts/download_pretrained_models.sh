#!/bin/bash

download_face() {
    mkdir -p pretrained_models/celeba256
    wget -O pretrained_models/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
    cd pretrained_models/celeba256
    unzip -o celeba-256.zip
    python split_model.py
}

download_ldm() {
    mkdir -p pretrained_models/LAION_text2img
    wget -O pretrained_models/LAION_text2img/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
    cd pretrained_models/LAION_text2img
    python split_model.py
}

download_sd1_5() {
    mkdir -p pretrained_models/SD1_5
    wget -O pretrained_models/SD1_5/model.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
    cd pretrained_models/SD1_5
    python split_model.py
}

download_sd2_1() {
    mkdir -p pretrained_models/SD2_1
    wget -O pretrained_models/SD2_1/model.ckpt https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-nonema-pruned.ckpt
    cd pretrained_models/SD2_1
    python split_model.py
}

download_all() {
    download_face
    cd ../..
    download_ldm
    cd ../..
    download_sd1_5
    cd ../..
    download_sd2_1
}

case $1 in
    "face")
        download_face
        ;;
    "ldm")
        download_ldm
        ;;
    "SD1_5")
        download_sd1_5
        ;;
    "SD2_1")
        download_sd2_1
        ;;
    "all")
        download_all
        ;;
    *)
        echo "Invalid argument. Usage: bash download.sh [face|ldm|SD1_5|SD2_1|all]"
        ;;
esac
