

# AAAI-2023-Code

```
@inproceedings{dong@2023aaai,
  title={Towards Reliable Item Sampling for Recommendation Evaluation},
  author={Dong Li and Ruoming Jin and Zhenming Liu and Bin Ren and Jing Gao and Zhi Liu},
  booktitle={Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI} 2023},
  publisher={{AAAI} Press},
  year={2023},
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

