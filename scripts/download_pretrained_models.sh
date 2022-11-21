mkdir -p pretrained_models/celeba256
mkdir -p pretrained_models/LAION_text2img
wget -O pretrained_models/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
wget -O pretrained_models/LAION_text2img/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

cd pretrained_models/celeba256
unzip -o celeba-256.zip
python split_model.py

cd ../LAION_text2img
python split_model.py