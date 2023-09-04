

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

