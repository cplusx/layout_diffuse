mkdir -p pretrained_models/celeba256
mkdir -p pretrained_models/LAION_text2img
mkdir -p pretrained_models/SD2_1
wget -O pretrained_models/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
wget -O pretrained_models/LAION_text2img/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
wget -O pretrained_models/SD2_1/model.ckpt https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-nonema-pruned.ckpt

cd pretrained_models/celeba256
unzip -o celeba-256.zip
python split_model.py

cd ../LAION_text2img
python split_model.py

cd ../SD2_1
python split_model.py