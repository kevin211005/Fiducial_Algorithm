# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:13:57 2020

@author: user
"""

import numpy as np 
from itertools import combinations
#import matplotlib.pyplot as plt
#import matplotlib  
#matplotlib.use('Agg') 
def avg_test(data,subject,beats):
    num_beat = int(len(data)/subject)
    combins = [c for c in  combinations(range(num_beat), 2)]
    avg_data = np.zeros((int(beats*subject),np.shape(data)[1],1))
    for sub in range(subject):
        for beat in range(beats):
            np.random.shuffle(data[sub*num_beat:(sub+1)*num_beat])
            avg_data[sub*beats+beat] = (data[sub*num_beat+combins[beat][0]]+data[sub*num_beat+combins[beat][1]])/2
    return avg_data
#%%
def min_curvature(data,L):
    result = np.zeros(len(data)-1)
    R_y = np.zeros((2,len(data)-1))
    if L == 1:    
        x = np.array(((len(data)-1),float(data[-1])))
        a = np.array((-(len(data)-1),float(data[0]-data[-1])))
    else:
        x = np.array(((0),float(data[0])))
        a = np.array(((len(data)-1),float(data[-1]-data[0])))
    for k in range(1,len(data)-1):
        y = np.array((k,float(data[k])))
        R_y[:,k] = y 
        c = y-x
        cos = np.dot(a,c)/(np.linalg.norm(a)*np.linalg.norm(c))
        if cos >=1:
            cos = 1            
        D = np.linalg.norm(c)*(np.sqrt(1-cos**2))
        result[k] = D
    indx = int(np.where(result == np.max(result))[0])
    rp = int(R_y[0,indx])
    rv = R_y[1,indx]
    return rp,rv
#%% 
def fiducial_fmap(data,style = 0):
    """ Geranl heart rate is about 1000      """
    """ General QRS range is about 80~120 ms  """
    QRS_range = int(np.shape(data)[1]*(100/1000))
    """ General P range is about 80ms        """
    P_range = int(np.shape(data)[1]*(80/1000))
    """ General T range is about 160ms        """
    T_range = int(np.shape(data)[1]*(160/1000))
    """ General PR range is about 200ms   """
    PR_range = int(np.shape(data)[1]*(200/1000))
    """ General QT range is about 200ms   """
    QT_range = int(np.shape(data)[1]*(300/1000))
    report = np.zeros((len(data),18))
    fiducial_feature = np.zeros((len(data),21))
    for i in range(len(data)):
        ## R detect Rv Rp 
        """R peak located in the center """
        center = int(np.shape(data)[1]/2)
        Rv = np.max(data[i][(center-QRS_range):(center+QRS_range)])
        Rp_interval = data[i][(center-QRS_range):(center+QRS_range)]
        Rp = int(np.argmax(Rp_interval))
        Rp = Rp + (center-QRS_range)
        
        ## Q detect Qv Qp
        Qv = np.min(data[i][(Rp-QRS_range):Rp])
        Qp_interval = data[i][(Rp-QRS_range):Rp]
        Qp = int(np.argmin(Qp_interval))
        Qp = Qp + (Rp-QRS_range)
        
        ## S detect Sv Sp
        Sv = np.min(data[i][Rp:(Rp+QRS_range)])
        Sp_interval = data[i][Rp:(Rp+QRS_range)]
        Sp = int(np.argmin(Sp_interval))
        Sp = Sp + Rp    
        
        ## P detect Pv Pp
        Pv = np.max(data[i][Qp-PR_range:Qp])
        Pp_interval = data[i][Qp-PR_range:Qp]
        Pp = int(np.argmax(Pp_interval))
        Pp = Pp + (Qp-PR_range) 
        
        ## T detect Tv Tp
        Tv = np.max(data[i][Sp:Sp+QT_range])
        Tp_interval = data[i][Sp:Sp+QT_range]
        Tp = int(np.argmax(Tp_interval))
        Tp = Sp+Tp

        # L' detect  Lp Lv
        #check range 
        lP_range = Pp-P_range
        while (lP_range<0):
            lP_range+=1
        L = 1
        lp,lv = min_curvature(data[i][lP_range:Pp],L)
        lp = lp+lP_range
        
        # t' detect  tp tv
        #check range 
        Tt_range = Tp+T_range
        while (Tt_range>np.shape(data)[1]):
            Tt_range-=1
        L = 0
        tp,tv = min_curvature(data[i][Tp:Tt_range],L)
        tp = tp + Tp
        
        # p' detect  pp pv
        #check range 
        Pp_range = Pp+P_range
        L = 0
        pp,pv = min_curvature(data[i][Pp:Pp_range],L)
        pp = pp + Pp
        
        # s' detect  sp sv
        #check range 
        sT_range = Tp-T_range
        L = 1
        sp,sv = min_curvature(data[i][sT_range:Tp],L)
        sp = sp + sT_range
        
        ## figure info 
        report[i,0] = lv
        report[i,1] = lp
        report[i,2] = Pv
        report[i,3] = Pp    
        report[i,4] = pv
        report[i,5] = pp
        report[i,6] = Qv
        report[i,7] = Qp   
        report[i,8] = Rv
        report[i,9] = Rp
        report[i,10] = Sv
        report[i,11] = Sp    
        report[i,12] = sv
        report[i,13] = sp
        report[i,14] = Tv
        report[i,15] = Tp           
        report[i,16] = tv
        report[i,17] = tp        
        """
        feature map =   [RQ,RS,RP,Rl,Rp,RT,Rs,Rt,lp,st,ST,PQ,PT,LQ,St 
                         Pl,PQ,RQ,RS,TS,Tt]
        """
        ## temporal 
        fiducial_feature[i,0] = np.abs(Rp-Qp)
        fiducial_feature[i,1] = np.abs(Rp-Sp)
        fiducial_feature[i,2] = np.abs(Rp-Pp)
        fiducial_feature[i,3] = np.abs(Rp-lp)
        fiducial_feature[i,4] = np.abs(Rp-pp)
        fiducial_feature[i,5] = np.abs(Rp-Tp)
        fiducial_feature[i,6] = np.abs(Rp-sp)
        fiducial_feature[i,7] = np.abs(Rp-tp)
        fiducial_feature[i,8] = np.abs(lp-pp)
        fiducial_feature[i,9] = np.abs(sp-tp)
        fiducial_feature[i,10] = np.abs(Sp-Tp)
        fiducial_feature[i,11] = np.abs(Pp-Qp)
        fiducial_feature[i,12] = np.abs(Pp-Tp)
        fiducial_feature[i,13] = np.abs(lp-Qp)
        fiducial_feature[i,14] = np.abs(Sp-tp)
        ## amplitude
        fiducial_feature[i,15] = np.abs(Pv-lv)
        fiducial_feature[i,16] = np.abs(Pv-Qv)
        fiducial_feature[i,17] = np.abs(Rv-Qv)
        fiducial_feature[i,18] = np.abs(Rv-Sv)
        fiducial_feature[i,19] = np.abs(Tv-Sv)
        fiducial_feature[i,20] = np.abs(Tv-tv)
    if style == 0:       
        return fiducial_feature
    elif style ==1:
        return fiducial_feature,report 
#%%
def Template_gen(data,subject):
    template = np.zeros((subject,np.shape(data)[1]))
    for sub in range (subject):
        for beat in range(int(len(data)/subject)):
            template[sub,:]+=data[sub*int(len(data)/subject)+beat,:]
    template = template/int(len(data)/subject)
    return template

def IR(test,template,style = 0):
    sub = int(len(template))
    beat = int(len(test)/sub)
    correct  = 0 
    dis_metrix = np.zeros((sub,len(test)))
    for i in range(len(test)):
        dis = np.linalg.norm(template-test[i],axis = 1)
        dis_metrix[:,i] = dis
    min_dis = np.argmin(dis_metrix,axis=0)
    for i in range(sub):
        for j in range(beat):
            if min_dis[i*beat+j] == i:
                correct+=1
    ir = (correct /len(test))*100
    if style ==1:
        return dis_metrix,min_dis,ir 
    elif style ==0:
        return ir
def Outlier_exclude(template,train,test,outlier,enroller_num,outlier_num,style=0):
    train_beat = int(len(train)/enroller_num)
    test_beat = int(len(test)/enroller_num)
    outlier_beat = int(len(outlier)/outlier_num)
    """
            sub1 sub2.....
    Q = q1|               |
        q3|               |
    """
    Q = np.zeros((2,enroller_num)) 
    dis_train = np.zeros((train_beat,enroller_num))
    for sub in range(enroller_num):
        for beat in range(train_beat):
            dis = np.linalg.norm(template[sub]-train[sub*train_beat+beat])
            dis_train[beat,sub] = dis
    Q[0,:] = np.percentile(dis_train,[25],axis=0)
    Q[1,:] = np.percentile(dis_train,[75],axis=0)  
    ##FPIR FNIR
    FPIR = []
    FNIR = []
    dis_test = np.zeros((enroller_num,len(test)))
    for i in range(len(test)):
        dis = np.linalg.norm(template-test[i],axis = 1)
        dis_test[:,i] = dis
    min_dis_test = np.argmin(dis_test,axis=0)
    #FNIR
    dis_outlier = np.zeros((enroller_num,len(outlier)))
    for i in range(len(outlier)):
        dis = np.linalg.norm(template-outlier[i],axis = 1)
        dis_outlier[:,i] = dis
    min_dis_outlier = np.argmin(dis_outlier,axis=0)
    for k in range(20):
        K = k*0.25
        correct = 0  
        fpir = 0
        for i in range(enroller_num):
            for j in range(test_beat):
                if min_dis_test[i*test_beat+j] == i:
                   index = min_dis_test[i*test_beat+j] 
                   if dis_test[index,i*test_beat+j] <= Q[1,i] + K*(Q[1,i]-Q[0,i]):
                       correct+=1
        for i in range(outlier_num):
            for j in range(outlier_beat):
               index = min_dis_outlier[i*outlier_beat+j] 
               if dis_outlier[index,i*outlier_beat+j] <= Q[1,index] + K*(Q[1,index]-Q[0,index]):
                   fpir+=1
        FNIR.append((len(test)-correct)/len(test)*100)
        FPIR.append(fpir/len(outlier)*100)
    return FNIR, FPIR   
#%% save figure 
#save_path = '/home/user/Desktop/DATA3/u105011242/reference/fiducial_test_fig/'
#for sub in range(1):
#    index =sub*30    
#    figure  = plt.figure()
#    for i in range(5): 
#        plt.plot(data[index+i])    
#        plt.scatter([report[index+i,1]],[report[index+i,0]], marker = 'o', c='m',s=80)
#        plt.scatter([report[index+i,3]],[report[index+i,2]], marker = 'o', c='k',s=80)
#        plt.scatter([report[index+i,5]],[report[index+i,4]], marker = 'o', c='g',s=80)
#        plt.scatter([report[index+i,7]],[report[index+i,6]], marker = 'o', c='r',s=80)
#        plt.scatter([report[index+i,9]],[report[index+i,8]], marker = 'o', c='k',s=80)
#        plt.scatter([report[index+i,11]],[report[index+i,10]], marker = 'o', c='r',s=80)
#        plt.scatter([report[index+i,13]],[report[index+i,12]], marker = 'o', c='g',s=80)
#        plt.scatter([report[index+i,15]],[report[index+i,14]], marker = 'o', c='k',s=80)
#        plt.scatter([report[index+i,17]],[report[index+i,16]], marker = 'o', c='m',s=80)
##    plt.savefig(save_path+'Person_'+str(sub+1)+'.png')