# Basic_CNNs_TensorFlow2
A tensorflow2 implementation of some basic CNNs.

## Networks included:
+ ShuffleNet_V2


## Train
1. Requirements:
+ Python >= 3.6
+ Tensorflow >= 2.4.0
+ tensorflow-addons>=0.12.0
2. To train the network on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——raw dataset
   |——feature
        class_name_0.jpg
        class_name_1.jpg
   |——label
        class_name_0.json
        class_name_1.json
```
3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set. The dataset directory will be like this:
 ```
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
 ```
4. Run **to_tfrecord.py** to generate tfrecord files.
5. Change the corresponding parameters in **config.py**.
6. Run **train.py** to start training.<br/>
## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.