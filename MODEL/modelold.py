#coding:utf-8
"""===============================================================================================
The Python code of GAN to realize 1-dim feature classification. The model we used includes DCGAN 
and WGAN.
---------------------------------------------------------------------------------------------------
Class: Model
Param: 	self.trainable = True, self.batch_size = 50, self.lr = 0.0002, self.mm = 0.5 ,
        self.z_dim = 128 ,self.EPOCH = 50 ,self.LAMBDA = 0.1 ,self.model = args.model ,self.dim = 1
        self.num_class = args.label_num+1 ,self.load_model = args.load_model ,self.Threshold = 0.80
        self.x_data = feature ,self.y_data = label ,self.train_rate = 0.8 ,self.feature_dim = 1024
        self.trainable = True               ----(BOOL) True for training, False for testting
        self.batch_size = 50                ----(int) batch size
        self.lr = 0.0002                    ----(float) learning rate
        self.mm = 0.5                       ----(float) momentum term for adam
        self.EPOCH = 5                      ----(int) Number of training rounds.
        self.LAMBDA = 0.1                   ----(float) parameter of WGAN-GP
        self.model = args.model             ----(string) Model Type, namely 'DCGAN' or 'WGAN'
        self.dim = 1                        ----(int) RGB is different with gray pic
        self.num_class = args.label_num+1   ----(int) num of classes
        self.load_model = args.load_model   ----(BOOL) True for load ckpt model and False for otherwise.
        self.Threshold = 0.80               ----(float) The lower-bound acc to save model
        self.x_data = feature               ----(numpy array) The inpuuted data
        self.y_data = label                 ----(numpy array) The outputed data
        self.train_rate = 0.8               ----(float) Proportion of training set.
        self.feature_dim = 1024             ----(float) The feature dim
        self.z_dim = self.feature_dim       ----(float) The dimension of noise z
---------------------------------------------------------------------------------------------------
Tip: None
---------------------------------------------------------------------------------------------------
Created on Sat Dec 12 15:07:26 2020
@author: 西电博巍(Bowei Wang)
Version: Ultimate
==============================================================================================="""
from warnings import filterwarnings
filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import os
import DLL.plot as plot
from DLL.utils import Eval,setup_logger,AucPlt1,AucPlt2
from MODEL.layers import lrelu,fc,conv1d,deconv1d
import DLL.save_images as save_img
import time
from tensorflow.contrib.slim import flatten
from math import log
from numpy import arange,vsplit
from numpy.random import shuffle
import logging


class Model(object):
    
    def __init__(self, 
                 args,
                 feature,
                 label,
                 add_fea,
                 add_label,
                 train_name,
                 batch_size = 50,
                 lr = 1e-4,
                 mm = 0.5,
                 EPOCH = 50,
                 LAMBDA = 0.1,
                 dim = 1,
                 Threshold = 0.00,
                 train_rate = 0.8,
                 feature_dim = 1024,
                 remark = "No remarks yet."):
        self.add_fea = add_fea
        try:
            self.add_label = self.OneHot(add_label)
        except AttributeError:
            self.add_label = None
        self.trainable = args.trainable
        self.batch_size = batch_size  # must be even number
        if args.load_model:
            self.batch_size = 1
        self.lr = lr
        self.mm = mm      # momentum term for adam
        self.EPOCH = EPOCH    # the number of max epoch
        self.LAMBDA = LAMBDA  # parameter of WGAN-GP
        self.model = args.model  # 'DCGAN' or 'WGAN'
        self.dim = dim       # RGB is different with gray pic
        self.num_class = args.label_num+1
        self.load_model = args.load_model
        self.Threshold = Threshold
        self.Threshold_Se = 0.0
        self.Threshold_Sp = 0.0
        self.Threshold_Mcc = 0.0
        self.x_data = feature
        self.depth_li = [32,64,128,64,32]
        #One Hot
        self.y_data = self.OneHot(label)
        self.train_rate = train_rate
        self.feature_dim = feature_dim #the feature dim
        self.z_dim = 128   # the dimension of noise z
        self.train_name = train_name
        self.build_model()  # initializer
        self.remark = remark
    
    def OneHot(self,label):
        label = label.reshape((-1,1))
        label2 = label+1
        label2[label2 == 2] = 0
        label = np.c_[label2,label]
        del label2
        return label
    
    @property
    def GetAllPara(self):
        return {'trainable': self.trainable,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'mm': self.mm,
                'EPOCH': self.EPOCH,
                'LAMBDA': self.LAMBDA,
                'model': self.model,
                'dim': self.dim,
                'num_class': self.num_class,
                'load_model': self.load_model,
                'Threshold': self.Threshold,
                'train_rate': self.train_rate,
                'feature_dim': self.feature_dim,
                'z_dim': self.z_dim,
                'remark': self.remark,
                'train_name':self.train_name,
                'help': ['trainable','batch_size','lr','mm','EPOCH','LAMBDA','model','dim','num_class','load_model',
                         'Threshold','train_rate','feature_dim','z_dim','remark','train_name']}


    def GetTrainGroup(self,x_data,y_data,train_rate = None,is_shuffle = True):
        if train_rate is None:
            train_rate = self.train_rate
        else:
            pass
        total_size = x_data.shape[0]
        total_index = arange(total_size)
        if is_shuffle: shuffle(total_index)
        #Split into train and  test
        train_index = total_index[:int(total_size*train_rate)]
        test_index = total_index[int(total_size*train_rate):]
        xtrain,ytrain = x_data[train_index],y_data[train_index]
        xtest,ytest = x_data[test_index],y_data[test_index]
        #Grouping
        train_size,test_size = xtrain.shape[0],xtest.shape[0]
        train_end = train_size if train_size%self.batch_size == 0 else (train_size//self.batch_size)*self.batch_size
        test_end = test_size if test_size%self.batch_size == 0 else (test_size//self.batch_size)*self.batch_size
        xtrain,ytrain,xtest,ytest = xtrain[:train_end],ytrain[:train_end],xtest[:test_end],ytest[:test_end]
        xtrain_li = vsplit(xtrain,train_end//self.batch_size)
        ytrain_li = vsplit(ytrain,train_end//self.batch_size)
        try:
            xtest_li = vsplit(xtest,test_end//self.batch_size)
            ytest_li = vsplit(ytest,test_end//self.batch_size)
        except ZeroDivisionError:
            xtest_li = None
            ytest_li = None
        return xtrain_li,ytrain_li,xtest_li,ytest_li

    def GrenerateTrainOp(self,loss,var_list):
        global_step = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(self.lr,global_step,self.batch_size,0.9,staircase = True)
        train_op = tf.train.AdamOptimizer(lr,beta1=self.mm)
        grads,v = zip(*train_op.compute_gradients(loss, var_list = var_list))
        grads,_ = tf.clip_by_global_norm(grads,5)
        train_op = train_op.apply_gradients(zip(grads,v),global_step = global_step)
        return train_op

    def build_model(self):
        # build  placeholders
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32,shape = [self.batch_size,self.feature_dim*1],name = 'D_Input')
        self.z = tf.placeholder(tf.float32,shape = [self.batch_size,self.z_dim],name = 'Noise')
        self.label = tf.placeholder(tf.float32,shape = [self.batch_size,self.num_class-1],name = 'Label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='Flag')
        self.flag2 = tf.placeholder(tf.float32, shape=[], name='Flag2')
        
        # define the network
        self.G_feature = self.generator('gen',self.z,reuse = False)
        print(self.G_feature)
        x_feature = tf.reshape(self.x, (self.batch_size, self.feature_dim,self.dim))
        d_in = tf.concat([x_feature,self.G_feature],axis = 0)
        
        self.D_logits_, self.D_out_ = self.discriminator('dis', d_in,keep_pro = 0.5, reuse=False)
 
        self.D_logits, self.D_logits_f = tf.split(self.D_logits_, [self.batch_size, self.batch_size], axis=0)
 
        d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')
 
        #Caculate the Supervised Loss
        batch_gl = tf.zeros_like(self.label, dtype=tf.float32)
        batchl_ = tf.concat([self.label, tf.zeros([self.batch_size, 1])], axis=1)
        batch_gl = tf.concat([batch_gl, tf.ones([self.batch_size, 1])], axis=1)
        batchl = tf.concat([batchl_, batch_gl], axis=0)*0.9  # one side label smoothing
        s_l = tf.losses.softmax_cross_entropy(onehot_labels=batchl, logits=self.D_logits_, label_smoothing=0)
        
        #Caculate the Unsupervised Loss
        s_logits_ = tf.nn.softmax(self.D_logits_)
        un_s = tf.reduce_sum(s_logits_[:self.batch_size, -1])/(tf.reduce_sum(s_logits_[:self.batch_size,:])) \
                + tf.reduce_sum(s_logits_[self.batch_size:,:-1])/tf.reduce_sum(s_logits_[self.batch_size:,:])
        
        #Caculate the Unsupervised Loss
        f_match = tf.constant(0., dtype=tf.float32)
        for i in range(len(self.depth_li)):
            d_layer, d_glayer = tf.split(self.D_out_[i], [self.batch_size, self.batch_size], axis=0)
            f_match += tf.nn.l2_loss(tf.subtract(d_layer, d_glayer))
        if self.model=='WGAN-GP':
            self.g_loss = tf.reduce_mean(self.D_logits_f[:,-1]) + f_match*0.01*self.flag2
            disc_cost = -tf.reduce_mean(self.D_logits_f[:,-1]) + tf.reduce_mean(self.D_logits[:,-1])
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            differences = self.G_feature - x_feature
            differences = tf.reshape(differences, (self.batch_size, self.feature_dim*self.dim))
            interpolates = x_feature + tf.reshape((alpha * differences), (self.batch_size, 1024, self.dim))
            D_logits, _ = self.discriminator('dis', interpolates, keep_pro = 0.5, reuse=True)
            gradients = tf.gradients(D_logits[:,-1], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.d_l_1, self.d_l_2, self.d_l_3 = disc_cost + self.LAMBDA * gradient_penalty, self.flag*s_l, (1-self.flag)*un_s
            self.d_loss = 0.1*self.d_l_1 + self.d_l_2 + self.d_l_3 + d_regular
        elif self.model=='DCGAN':
            self.d_loss_real = tf.losses.softmax_cross_entropy(onehot_labels=batchl_*0.9, logits=self.D_logits)
            self.d_loss_fake = tf.losses.softmax_cross_entropy(onehot_labels=batch_gl*0.9, logits=self.D_logits_f)
            self.g_loss = self.d_loss_fake + f_match*0.01*self.flag2
            self.d_l_1, self.d_l_2, self.d_l_3 = self.d_loss_fake + self.d_loss_real, self.flag*s_l, (1-self.flag)*un_s
            self.d_loss = self.d_l_1 + self.d_l_2 + self.d_l_3 + d_regular
        else:
            print('model must be DCGAN or WGAN-GP!')
            return
        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]
 
        if self.model == 'DCGAN':
            # self.opt_d = self.GrenerateTrainOp(self.d_loss, var_list = d_vars)
            # self.opt_g = self.GrenerateTrainOp(self.g_loss, var_list = g_vars)
            self.opt_d = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'WGAN-GP':
            self.opt_d = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        else:
            print ('model can only be "DCGAN","WGAN_GP" !')
            return
        test_logits, _ = self.discriminator('dis', self.x,keep_pro = 1, reuse=True)
        self.test_logits = tf.nn.softmax(test_logits)
        self.pre_logits = tf.argmax(self.test_logits, axis=1)
        self.prediction = tf.nn.in_top_k(self.test_logits, tf.argmax(batchl_, axis=1), 1)
  
        if not self.load_model:
            pass
        elif self.load_model:
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, os.getcwd()+'/TRAINED_MODEL/model.ckpt')
            print('model load done')

    def train(self):
        
        setup_logger('base','./TRAINED_MODEL/','train_on_'+self.train_name, level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        #xtrain_li,ytrain_li,_,_ = self.GetTrainGroup(self.x_data,self.y_data)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        
        wirter = tf.summary.FileWriter('./LOGS/',self.sess.graph)

        if not os.path.exists('TRAINED_MODEL'):
            os.mkdir('TRAINED_MODEL')
        if not os.path.exists('GEN_PICTURE'):
            os.mkdir('GEN_PICTURE')
        noise = np.random.normal(-1, 1, [self.batch_size, self.z_dim])
        logger.info('training.....................')
        for epoch in range(self.EPOCH):
            
            xtrain_li,ytrain_li,xtest_li,ytest_li = self.GetTrainGroup(self.x_data,self.y_data)
            
            iters = len(xtrain_li)
            flag2 = 1   if epoch>10 else 0
            for idx,(batchx,batchl) in enumerate(zip(xtrain_li,ytrain_li)):
                start_t = time.time()
                flag = 1 if idx<4 else 0 # set we use 500 train data with label.
                g_opt = [self.opt_g, self.g_loss]
                d_opt = [self.opt_d, self.d_loss, self.d_l_1, self.d_l_2, self.d_l_3]
                feed = {self.x:batchx, self.z:noise, self.label:batchl, self.flag:flag, self.flag2:flag2}
                # Update the Discrimater One Times
                _, loss_d, d1,d2,d3 = self.sess.run(d_opt, feed_dict=feed)
                # Update the Generator One Time
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                logger.info( ("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f, d1:%4f, d2:%4f, d3:%4f flag:%d"%
                        (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d, loss_g,d1,d2,d3,flag )) )
                plot.plot('d_loss', loss_d)
                plot.plot('g_loss', loss_g)
                if ((idx+1) % 100) == 0:  # flush plot picture per 1000 iters
                    plot.flush()
                plot.tick()
 
                if (idx+1)%iters==0:
                    print ('images saving............')
                    img = self.sess.run(self.G_feature, feed_dict=feed)
                    save_img.save_images(img, os.getcwd()+'/GEN_PICTURE/'+'sample{}_{}.jpg'\
                                         .format(epoch, (idx+1)/iters))
                    print('images save done')            
            Acc,Se,Sp,Mcc,train_label,train_Score = self.test(xtest_li,ytest_li,mark = "Feature n' Label")
            xadd_test_li,yadd_test_li,_,_ = self.GetTrainGroup(self.add_fea,self.add_label,train_rate=1,is_shuffle = False)
            addAcc,addSe,addSp,addMcc,test_label,test_Score = self.test(xadd_test_li,yadd_test_li,mark = "Additional Feature n' Additional Label")
            
            
            plot.plot('test acc', Acc)
            plot.flush()
            plot.tick()
            message = 'Test Acc:{}, Test Se:{}, Test Sp:{}, Test Mcc:{}.\n'.format(Acc,Se,Sp,Mcc) + \
                      'Threshold:%3f\n'%(self.Threshold) + \
                      'Add acc:{}, Add Se:{}, Add Sp:{}, Add Mcc:{}.\n'.format(addAcc,addSe,addSp,addMcc)
            logger.info(message)
            if addAcc > self.Threshold:
                AucPlt2('Auc Plot',train_label,train_Score,test_label,test_Score)
                logger.info('model saving..............')
                path = os.getcwd() + '/TRAINED_MODEL'
                save_path = os.path.join(path, "model.ckpt")
                self.saver.save(self.sess, save_path=save_path)
                self.Threshold = addAcc
                self.Threshold_Se = addSe
                self.Threshold_Sp = addSp
                self.Threshold_Mcc = addMcc
        wirter.close()
        message = 'Train End at'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+\
            'The Final Threshold Is: \n' + \
            'Add acc:{}, Add Se:{}, Add Sp:{}, Add Mcc:{}.\n'.format(self.Threshold,self.Threshold_Se,self.Threshold_Sp,self.Threshold_Mcc)
        logger.info(message)
        self.sess.close()
        return
            
    def generator(self,name,noise,reuse):
        '''4__0__>16__1__>64__2__>256__3__>1024__4__>4096...'''
        with tf.variable_scope(name,reuse = reuse):
            gen_layers = int(log(self.feature_dim)//log(4))
            l = self.batch_size
            y = fc('g_fc',noise,2*2*64)
            y = tf.reshape(y, [-1, 2 * 2, 64])
            y = lrelu(y)
            for i in range(1,gen_layers):
                f_dim,f_depth = 4**(i+1),2**(gen_layers-1-i)
                y = deconv1d('g_dconv{}'.format(i), y, 5, outshape = [l, f_dim, f_depth])
                y = lrelu(y)
            y = tf.reshape(y,[l,-1])
            #y = tf.squeeze(y)
            y = fc('g_fc2',y,self.feature_dim)
            y = lrelu(y)
            y = fc('g_fc3',y,self.feature_dim)
            return tf.reshape(y,(l,self.feature_dim,self.dim))
        

    def discriminator(self,name,inputs,keep_pro,reuse):
        l = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (l,self.feature_dim,self.dim))
        with tf.variable_scope(name,reuse=reuse):
            out = []
            y = inputs
            for i,n in enumerate(self.depth_li):
                y = conv1d('d_con{}'.format(i),y,3, n, stride=1, padding='SAME')
                y = lrelu(self.bn('d_bn{}'.format(i),y))
                out += [y]
            y = flatten(y)
            y = fc('d_fc2', y, self.num_class)
            return y, out

    def bn(self, name, input):
        val = tf.contrib.layers.batch_norm(input, decay=0.9,
                                           updates_collections=None,
                                           epsilon=1e-5,
                                           scale=True,
                                           is_training=True,
                                           scope=name)
        return val
    
    def test(self,xtest_li,ytest_li,mark = ''):
        print('Testing on set '+mark+'................')
        logits_li = []
        test_label_li = []
        pre_score_li = []
        for i,(testx, textl) in enumerate(zip(xtest_li,ytest_li)):
            prediction,test_logits = self.sess.run([self.prediction,self.test_logits], feed_dict={self.x:testx, self.label:textl})
            test_logits = test_logits[:,0:2]
            test_label = np.argmax(textl,axis = 1)
            test_label_li += [test_label]
            pre_logits = np.argmax(test_logits,axis = 1)
            pre_score = test_logits[:,1]/test_logits.sum(axis = 1)
            pre_score_li += [pre_score]
            logits_li += [pre_logits]
        print(np.hstack(logits_li).shape)
        Acc,Se,Sp,Mcc = Eval(np.hstack(logits_li),np.hstack(test_label_li))
        AucPlt1(name = 'Auc Curve '+mark,label = np.hstack(test_label_li),Score = np.hstack(pre_score_li))
        return Acc,Se,Sp,Mcc,np.hstack(test_label_li),np.hstack(pre_score_li)
    
    def TestNewSet(self):
        print('testing................')
        xtrain_li,ytrain_li,_,_ = self.GetTrainGroup(self.x_data,self.y_data,train_rate=1,is_shuffle = False)
        Acc,Se,Sp,Mcc,train_label,train_Score = self.test(xtrain_li,ytrain_li,mark = "Feature n' Label")
        return Acc,Se,Sp,Mcc,train_label,train_Score



