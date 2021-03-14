# iPro-GAN(iPro-GAN: A novel model based on generative adversarial learning
method for identifying promoters and their strength )
===============================================================================
#Description
The Python code of iPro-GAN: A novel model based on generative adversarial learning 
method for identifying promoters and their strength which realizes 1-dim feature
classification. 
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
#Tip: None
-------------------------------------------------------------------------------
#Date
Created on Sat Dec 12 15:07:26 2020
#@author: 西电博巍 (Bowei Wang)
#Version: Ultimate
-------------------------------------------------------------------------------
# usage 
The Tensorflow vision is 1.8.0 and we use python 3.6.7 to implement our code.
## train 
`python __main__.py --model=DCGAN --trainable=True --load_model=False --label_num=2
## test 
`python __main__.py --model=DCGAN  --trainable=False  --load_model=True
## Additionally
If U wanna see whole computing graph, input tensorboard --logdir='./LOGS/'
'==============================================================================
