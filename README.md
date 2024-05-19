

# Reference

```
@inproceedings{li@2020kdd,
  author = {Li, Dong and Jin, Ruoming and Gao, Jing and Liu, Zhi},
  title = {On Sampling Top-K Recommendation Evaluation},
  year = {2020},
  isbn = {9781450379984},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3394486.3403262},
  doi = {10.1145/3394486.3403262},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages = {2114–2124},
  numpages = {11},
  keywords = {recall, top-k, recommender systems, evaluation metric, hit ratio},
  location = {Virtual Event, CA, USA},
  series = {KDD '20}
}

@article{li@2023aaai,
  title={Towards Reliable Item Sampling for Recommendation Evaluation},
  volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/25561},
  DOI={10.1609/aaai.v37i4.25561},
  number={4},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Li, Dong and Jin, Ruoming and Liu, Zhenming and Ren, Bin and Gao, Jing and Liu, Zhi},
  year={2023}, month={Jun.}, pages={4409-4416}
}

@article{jin@2021aaai,
  title={On Estimating Recommendation Evaluation Metrics under Sampling},
  volume={35},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/16537},
  DOI={10.1609/aaai.v35i5.16537},
  number={5},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Jin, Ruoming and Li, Dong and Mudrak, Benjamin and Gao, Jing and Liu, Zhi},
  year={2021}, month={May}, pages={4147-4154}
}


```

### 1. Data Process 

generate 100 repeats test dataset with fix sample set size (n = 100 for example)

```console
cd ./data
python fix_sampling.py
```

Afterwards,  it would generate a *./fix_sample_100* folder contains 100 different test sets for each dataset.

### 2. Train recommendation models and generate ranks

```
cd ./models
```

Train the models, for example run *NeuMF_train.ipynb*.

After the model is trained, we are able to generate the rank for the test item among all items and among sampled set. run *NeuMF_repeat.ipynb*

Consequently, it would generate a *./fix_sample_100* folder contains 100 different rank files for each dataset.

### 3. Estimate the `P(R)` by estimators

```condole
cd ./estimators
```

Run *MLE.ipynb* for instance to estimate the `P(R)` according to the rank files based on the sample set. The outputs would be saved in *../save_PR* folder

### 4. Quantify the performance of the estimators 

*PICK_WINNER*, *Relative_Error* files are used to generate the final (table) results in our paper. 

The output file is stored in *./table/results* folder. 

Noting that limited by the file size, we did not put all output files here. 

# Adaptive Sampling Estimation

### 1. When to use

* Test/evaluation stage, not training stage. Adaptive sampling estimation is the same as a normal evaluation method that compute metrics for different models and evaluate/compare their performance

* When there are too many items (such as over millions), computing a specific rank of an item among all items are significant inefficient.


### 2. Why use
* Global Evaluation

  Assume there is a user defined rank function (after model is trained): 

```math
R^u_i = f(u, i, I/i)
```
where $u$ is the user id, $i$ is the test item id, $I$ is the total set of items. $R^u_i$ is the final rank.

* Sampling Evaluation

```math
r^u_i = f(u, i, I_s)
```
$I_s$ is a sample set of items

Sometimes $R^u_i$ is too much resources consuming, we have to rely on sampling-based evaluation. The issue is sampling-based evaluation can not correctly reflect the models' performance as we expected according to [KDD 2020 best paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226). Intuitively, $Recall@10$ of a model in sampling-based evaluation can be approximate to $Recall@1000$ in global estimation while the top-1000 is not really what we want (ref [ our KDD2020 paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403262)).

Adaptive sampling can help rectify the issue with given only sapling rank $r^u_i$ to estimate its global $R^u_i$ and compute metric effectively.
### 3. How to use

3.1 use 'adaptive/adaptive_sampling.py' to obtain the sample rank

3.2 use 'adaptive/adaptive_estimator.py' to estimate the global rank (distribution)

As long as PR is obtained from 3.2, one can use 'NDCG_K' from 'estimator.utils.py' to approximate global NDCG metric, etc.