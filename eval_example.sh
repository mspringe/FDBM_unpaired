python -m src.MAfBM.eval_bid --pi0 MNIST --pi1 EMNIST \
        --precision float32\
        --arch DiT_S_2 \
        --sqrt_eps 1 --K 5 \
        --H  0.5 \
        --MAfBM_norm True \
        --batch_size 128 \
        --micro_batch_size 128 \
        --ckpt_pretraining  \
        --log_dir logs/ --out_dir outputs/H0.5"
