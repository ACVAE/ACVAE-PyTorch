# ACVAE-PyTorch
PyTorch Code for Adversarial and Contrastive AutoEncoder for Sequential Recommendation.


## Usage
- python 3.6+
- PyTorch
- tqdm
- tensorboardX
- numpy

Run `train.py`:

```
python3 train.py
```

The dataset is set to `ml-1m` by default. You can change it by setting the `hyper_params` in `train.py`. For the convenience of reproduction, we provide 3 preprocessed datasets: `ml-latest`, `ml-1m` and `ml-10m`. All of the lines in the datasets are formatted as `[USER_ID] [ITEM_ID]` ordered by interaction timestamps.

If you want to train this model on your own datasets, you can save your preprocessed dataset files under `model_dat/`. You also need to add one item in `dataset_info.json`, which contains the information of the count of users and items as well as the `seq_len` to use in the model.
