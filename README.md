# IAB label classification finetuned on pretrained audio neural networks (PANNs)

IAB label classification is a task to classify audio clips into different IAB labels (defined here: https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/main/Content%20Taxonomies/Content%20Taxonomy%203.0.tsv). We used the Audioset dataset to train this model; more information on how we created the dataset can be found here: https://docs.google.com/document/d/1pQ1vQdil9zjRclzEl9cMfUElAkMGtnfrI-ooycJaLHk/edit?usp=sharing. In this codebase, we fine-tune PANNs based on how this model was fine tuned: https://github.com/qiuqiangkong/panns_transfer_to_gtzan/tree/master to build an audio clip classification system.

View our full report here on how we created the dataset and how we created our model: https://docs.google.com/presentation/d/10LhEUxOM6hpeT8t5M1WKPEtvoh_ltfPXeEt6nOOtCw4/edit?usp=sharing

**1. Requirements** 

python 3.6 + pytorch 1.0

**2. Then simply run:**

$ Run the bash script ./runme.sh

Or run the commands in runme.sh line by line. The commands includes:

(1) Modify the paths of dataset and your workspace

(2) Extract features

(3) Train model

## Model
A 14-layer CNN of PANNs is fine-tuned. We use 10-fold cross validation for IAB label classification. That is, 900 audio clips are used for training, and 100 audio clips are used for validation.

## Citation

[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." arXiv preprint arXiv:1912.10211 (2019).
