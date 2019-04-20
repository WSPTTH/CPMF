# CPMF

Implementation of "CPMF: A Collective Pairwise Matrix Factorization Model for Upcoming Event Recommendation".

If you find this method helpful for your research, please cite this paper:

```latex
@inproceedings{LiuZWXHG17,
  author    = {Chun{-}Yi Liu and
               Chuan Zhou and
               Jia Wu and
               Hongtao Xie and
               Yue Hu and
               Li Guo},
  title     = {{CPMF:} {A} collective pairwise matrix factorization model for upcoming event recommendation},
  booktitle = {2017 International Joint Conference on Neural Networks, {IJCNN} 2017},
  pages     = {1532--1539},
  year      = {2017}
}
```

---

### Requirement

* python >= 3.4
* numpy
* tqdm
* sklearn

----

### Dataset

The dataset can be downloaded from [large network](http://www.largenetwork.org/ebsn). Or connect the author of the paper "Event-based Social Networks: Linking the Online and Ofﬂine Social Worlds".

----

### How to use

#### Step 1. Extract the City Dataset

We extract the dataset of a city from the original dataset as follows:

```bash
python main.py --mode=prepro --pre_config=xxxx
```

The `pre_config` need to be specified. It is a `.json` file.

| keys         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `save_base`  | `str`. The path to save the preprocessed data. The real path will be the `save_base/city_name`.  The `city_name` is pre-defined in the `preprocess.py`. |
| `radius`     | `int`. The radius (kilometer) of a city. The city is treated as a cycle and the user/event within the cycle is treated as part of the city dataset. The centers of cities are pre-defined in the `preprocess.py`. |
| `wash`       | `int`. The parameter is to filter the users who attended less than `wash` events. |
| `user_file`  | The user location file. (`user_lon_lat.csv`)                 |
| `event_file` | The event location file. (`event_lon_lat.csv`)               |
| `ue_file`    | The file of  user-event pairs. (`user_event.csv`)            |
| `ug_file`    | The file of  user-group pairs. (`user_group.csv`)            |
| `ef_file`    | The file of  group-event pairs. (`event_group.csv`)          |

The process will generate several file in the saved path, including `user.loc`, `event.loc`, `ue.pair`, `ug.pair`, `eg.pair` and `info.json`.

#### Step 2: Train the model

Then, we can train our model with:

```bash
python main.py --mode=run --city_info_path=xxxxx
```

The `city_info_path` is the path of the generated `info.json`.  The input total datasets will first divide into a training dataset and a evaluation dataset. The saved paths can be specified by `train_path` and `dev_path` respectively. The saved files are both `.json` files. The ratio of the evaluation dataset is specified by `test_ratio`. 

If there exists both the `train_path` and `dev_path`, the pre-divided training and evaluation datasets will be loaded to avoid re-split.

Then, the model will be trained, and we can observe the performance of the training and evaluation dataset. And the model is auto saved in the training process. Note only **one** model will be saved, and the user should monitor the performance and manually conduct the early stopping.

There are also arguments to set the hyper-parameter of the model.and the introduction can be find in the `main.py`.

#### Step 3: Evaluation.

Given the evaluation dataset, we can conduct the evaluation with:

```bash
python main.py --mode=eval --dev_path=xxxxx
```

The `dev_path` is the path of the evaluation file.

----

### Disclaimer

The original codes were lost. I have attempted to re-create them as faithfully as possible but there may be some issues. If you find any bugs, please report them to me.