# AD-TUNING: An Adaptive CHILD-TUNING Approach to Efficient Hyperparameter Optimization of Child Networks for Speech Processing Tasks in the SUPERB Benchmark

### Abstract
AD-TUNING is an adaptive CHILD-TUNING approach for hyperparameter tuning of child networks. To address the issue of selecting an optimal hyperparameter set P , 
which often varies for different tasks in CHILD-TUNING, we first analyze 
the distribution of parameter importance to ascertain the range of P . Next, we propose a simple yet
efficient early-stop algorithm to select the appropriate child network from different sizes for various speech tasks. When evaluated on seven speech processing tasks in the SUPERB benchmark, our proposed framework only requires fine-tuning less
than 0.1%∼10% of pretrained model parameters for each task
to achieve state-of-the-art results in most of the tasks. For instance, the DER of the speaker diarization task is 9.22% relatively lower than the previously reported best results. Other
benchmark results are also very competitive. 

### Pipeline
![2023-05-19_00-06](https://github.com/liyunlongaaa/AD-TUNING/assets/49556860/20c2880d-ab89-44a2-a8ee-f6e7f62b5201)

### Prerequisites 

```
git clone https://github.com/liyunlongaaa/AD-TUNING.git
cd AD-TUNING
conda create -n  ad_tuning python=3.10
conda activate ad_tuning
pip install -e ".[all]"
```

### Training and Inference 
```
cd s3prl
bash run.sh > log.txt
```

More information (about data, config, training and inference) can be refered to [here](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md)
