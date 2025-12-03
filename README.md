# Fractional Schrödinger Bridge Flow


## install environment

Make sure you have cuda 12.x installed on your machine

`pip install -r requirements.txt`


## download and use data

MNIST & EMNIST datasets will be downloaded automatically, if they don't exist already.


You'll have to download AFHQ manually:

```
URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=./afhq.zip
mkdir -p ./datasets/AFHQ
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/AFHQ
mv ./datasets/AFHQ/afhq/* ./datasets/AFHQ/
rmdir ./datasets/AFHQ/afhq/
rm $ZIP_FILE
```


For latent space modelling be sure to generate your own latents and save them as numpy arrays (see `src/data/latent_datasets`)


## run training

Exemplary command MNIST &harr; EMNIST:


```
python -m src.train_MAfBM --progbar --pi0 MNIST --pi1 EMNIST \
    --precision float32 \
    --arch DiT_S_2 \
    --log_interval 2500 \
    --checkpoint_interval 5000 \
    --pretraining_steps 40000 \
    --finetuning_steps 10000 \
    --lr 0.0001
```


## generate samples

```
python -m src.MAfBM.eval_bid --pi0 MNIST --pi1 EMNIST \
    --precision float32\
    --arch DiT_S_2 \
    --sqrt_eps 1 --K 5 \
    --H  0.5 \
    --MAfBM_norm True \
    --batch_size 128 \
    --micro_batch_size 128 \
    --ckpt_pretraining <path to your checkpoint> \
    --log_dir logs/ --out_dir outputs/H0.5
```
