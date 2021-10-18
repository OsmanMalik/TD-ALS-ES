# More Efficient Sampling for Tensor Decomposition

This repo contains code used in the experiments in the paper 
> Osman Asif Malik. *More Efficient Sampling for Tensor Decomposition*. arXiv:2110.07631

The paper is available for download on [arXiv](https://arxiv.org/abs/2110.07631).
The perhaps somewhat cryptic repo name stands for "**T**ensor **D**ecomposition via **A**lternating **L**east **S**quares with **E**fficient **S**ampling."

## Referencing this code

If you use this code in any of your own work, please reference our paper:

```
@misc{malik2021efficient,
      title={More Efficient Sampling for Tensor Decomposition}, 
      author={Osman Asif Malik},
      year={2021},
      eprint={2110.07631},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

I have done my best to include relevant references in the code and this README, so please also cite those works as appropriate.

## Some further details

Some further details on the code follow below.

- The functions `cp_als_es.m` and `tr_als_es.m` are implementations of the proposed methods for CP and tensor ring decomposition, respectively. 
The recursive sketching and sampling procedures are implemented in `recursive_sketch_CP.m` and `draw_samples_CP.m` for the CP decomposition, and in `recursive_sketch_TR.m` and `draw_samples_TR.m` for the tensor ring decomposition.
- The scripts `experiment_1.m` and `experiment_2.m` were used to create the tables in the "sampling distribution comparison" experiments.
Portions of these scripts were also used to create the plots in the supplementary material.
- The script `experiment_3.m` was used for the feature extraction experiment.
- The script `test_1.m` compares for a CP decomposition least squares problem the exact leverage score sampling distribution, our estimated distribution, and the empirical distribution when sampling according to the estimated distribution using our sampling scheme.
The script `test_2.m` is the same but for the tensor ring decomposition.

## Requirements

This code requires [Tensor Toolbox](https://gitlab.com/tensors/tensor_toolbox) for Matlab. 
It also requires some functionality from the repo [tr-als-sampled](https://github.com/OsmanMalik/tr-als-sampled). 
Make sure that the functions in both are accessible before running the code in this repo.

## Compiling MEX functions

The `.c` files in this repo need to be compiled. 
This is easy to do (provided that the [appropriate compilers](https://www.mathworks.com/help/matlab/matlab_external/choose-c-or-c-compilers.html) are installed) via the following commands in Matlab:
```
>> mex countSketch.c
>> mex mt19937ar.c
``` 

## Other software included in this repo

- **Mersenne Twister C code.** 
This code is available for download [here](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/emt19937ar.html) and was authored by Takuji Nishimura and Makoto Matsumoto. 
The code, including the relevant license statement, is in the file `mt19937ar.c`. 
We modified this code so that it can be called from within Matlab via the MEX interface. 
- **Sketching codes.**
The files `countSketch.c` and `TensorSketch.m` were copied over from the [countsketch-matrix-tensor-id](https://github.com/OsmanMalik/countsketch-matrix-tensor-id) repo for convenience.


## Author contact information

Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on the paper. I can be reached at `oamalik (at) lbl (dot) gov` or at `osman (dot) malik (at) colorado (dot) edu`. 
