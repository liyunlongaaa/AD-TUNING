# AD-TUNING: An Adaptive CHILD-TUNING Approach to Efficient Hyperparameter Optimization of Child Networks for Speech Processing Tasks in the SUPERB Benchmark

### Abstract
we propose AD-TUNING, an adaptive CHILD-TUNING approach for hyperparameter tuning of child networks. To address the issue of selecting an optimal hyperparameter set P , 
which often varies for different tasks in CHILD-TUNING, we first analyze 
the distribution of parameter importance to ascertain the range of P . Next, we propose a simple yet
efficient early-stop algorithm to select the appropriate child net-
work from different sizes for various speech tasks. When eval-
uated on seven speech processing tasks in the SUPERB bench-
mark, our proposed framework only requires fine-tuning less
than 0.1%∼10% of pre-trained model parameters for each task
to achieve state-of-the-art results in most of the tasks. For in-
stance, the DER of the speaker diarization task is 9.22% rel-
atively lower than the previously reported best results. Other
benchmark results are also very competitive. 

### Pipeline
![2023-05-19_00-06](https://github.com/liyunlongaaa/AD-TUNING/assets/49556860/20c2880d-ab89-44a2-a8ee-f6e7f62b5201)

### Prerequisites 

```
git clone https://github.com/liyunlongaaa/AD-TUNING.git
cd AD-TUNING
conda create -n  ad_tuning python=3.10
pip install -e "[.all]"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


```

### Training and Inference 
```
cd s3prl
bash run.sh > log.txt
```

More information on training and inference can be refered to [s3prl](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md)
