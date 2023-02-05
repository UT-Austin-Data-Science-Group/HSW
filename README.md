# Hierarchical Sliced Wasserstein distance
Python3 implementation of the papers [Hierarchical Sliced Wasserstein distance](https://arxiv.org/abs/2209.13570)

Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2023hierarchical,
  title={Hierarchical Sliced Wasserstein Distance},
  author={Khai Nguyen and Tongzheng Ren and Huy Nguyen and Litu Rout and Tan Nguyen and Nhat Ho},
  journal={Advances in Neural Information Processing Systems},
  year={2023},
  pdf={https://arxiv.org/pdf/2209.13570.pdf},
  code={https://github.com/UT-Austin-Data-Science-Group/HSW}
}
```

Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io). README is on updating process.

## Requirement

The code is implemented with Python (3.8.8) and Pytorch (1.9.0).

## What is included?
* (Hierarchical) Sliced Wasserstein Generators

## (Hierarchical) Sliced Wasserstein Generators
### Code organization
* cfg.py : this file contains arguments for training.
* datasets.py : this file implements dataloaders.
* functions.py : this file implements training functions.
* trainsw.py : this file is the main file for running SW.
* trainmaxsw.py : this file is the main file for running Max-SW.
* trainhsw.py : this file is the main file for running HSW.
* trainmaxhsw.py : this file is the main file for running Max-HSW.
* models : this folder contains neural networks architecture.
* utils : this folder contains implementation of fid score and Inception score.
* fid_stat : this folder contains statistic files for fID score.


### Main path arguments
* --dataset : type of dataset {"cifar10","tinyimagenet","celeba""}
* --img_size : size of images
* --dis_bs : size of mini-batches
* --model : "sngan_{dataset}"
*. --eval_batch_size : batchsize for computing FID
* --L : number of projections for (H) SW
* --k : number of bottleneck projections for (Max-)HSW
* --s_lr : slice learning rate (for Max-SW and Max-HSW)
* --s_max_iter : max iterations of gradient update (for Max-SW and Max-HSW)

### Example

```
 python trainhsw.py -gen_bs 128 -dis_bs 128 --data_path ./data --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --Ls 70,2000 --exp_name hsw --random_seed 1
```

## Acknowledgment
The structure of this repo is largely based on [sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch) and [CSW](https://github.com/UT-Austin-Data-Science-Group/CSW).