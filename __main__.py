#coding:utf-8

import numpy as np
from MODEL.model import Model
from warnings import filterwarnings
filterwarnings('ignore')
import argparse
from DLL.utils import LoadData

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DCGAN', help='DCGAN or WGAN-GP')
parser.add_argument('--trainable', type=bool, default= True,help='True for train and False for test.')
parser.add_argument('--load_model', type=bool, default=False, help='True for load ckpt model and False for otherwise.')
parser.add_argument('--label_num', type=int, default=2, help='the num of labled features we use.')
args = parser.parse_args()

def main():
    fea_path = './DATABASE/Feature.npy'
    label_path = './DATABASE/label.npy'
    feature = np.load(fea_path)
    label = np.load(label_path)
    if args.trainable:
        model = Model(args,
                      feature = feature,
                      label = label,
                      feature_dim = feature.shape[1],
                      train_name = 'Feature',
                      remark = 'Layer I')
        model.train()
        return "-----<The training process has been complished.>-----"
    else:
        model = Model(args,
                      feature = feature,
                      label = label,
                      feature_dim = feature.shape[1],
                      train_name = None)
        Acc,Se,Sp,Mcc,train_label,train_Score= model.TestNewSet()
        print(Acc,Se,Sp,Mcc)
        return "-----<The test Acc:{:.4f}, test Se:{:.4f}, test Sp:{:.4f}, test Mcc:{:.4f}>-----".format(Acc,Se,Sp,Mcc)

if __name__ == '__main__':
    
    print(main())
    
    
    
    