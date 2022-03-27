# Chinese Character Recognition DeepLearning Project

The data we have used is "CASIA-HWDB".


reference: http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html


Kaggle link to data: https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets

tfhub.py: python script to train a classfication model dealing with Chinese character data (3000 different characters).

Siamese.ipynb: train a convolutional siamese network on Chinese character data; train a VAE model for Chinese character generation (3000 different characters).

Flatten_Siamese_Network.ipynb: train a flatten siamese network on Chinese character data (200 different characters).

images folder: results of Chinese character generation with BigGAN at different training steps.

mini_lucid_tf2 folder: used to visualize the hidden convolutional layers.

For training BigGAN, we have used the code from: https://github.com/ajbrock/BigGAN-PyTorch.
