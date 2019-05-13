import numpy as np
import math
import pandas as pd
import ROOT

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Lambda,multiply,Multiply,RepeatVector,Flatten,Concatenate,Dropout
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

class ToyDetectors:
    
    def __init__(self):
        mass_e = 0.000511
        mass_mu = 0.105
        mass_pi = 0.139
        mass_K = 0.494
        mass_p = 0.938
        self.masses = [ mass_e,mass_mu, mass_pi, mass_K , mass_p]
    
    def BetheBlochAlephNP(self,lnbg,kp1=0.76176e-1,kp2=10.632,kp3=0.13279e-4,kp4=1.8631,kp5=1.9479):
        bg   = np.exp(lnbg)
        beta = bg/np.sqrt(1.+ bg*bg)
        aa   = np.exp(kp4*np.log(beta))
        bb   = np.exp(-kp5*np.log(bg))
        bb   = np.log(kp3+bb)
        return (kp2-aa-bb)*kp1/aa

    def BetheBlochGeantNP(self,lnbg,kp0=2.33,kp1=0.20,kp2=3.00,kp3=173e-9,kp4=0.49848):
        bg   = np.exp(lnbg)
        mK  = 0.307075e-3
        me  = 0.511e-3
        rho = kp0
        x0  = kp1*2.303
        x1  = kp2*2.303
        mI  = kp3
        mZA = kp4
        bg2 = bg*bg
        maxT= 2*me*bg2

        d2 = 0.
        x = np.log(bg)
        lhwI = np.log(28.816*1e-9*np.sqrt(rho*mZA)/mI)
        if x>x1 :
            d2 = lhwI + x - 0.5
        else :
            if x>x0:
                r = (x1-x)/(x1-x0)
                d2 = lhwI + x - 0.5 + (0.5 - lhwI - x0)*r*r*r

        return mK*mZA*(1+bg2)/bg2*(0.5*np.log(2*me*bg2*maxT/(mI*mI)) - bg2/(1+bg2) - d2)

    def BetheBlochSolidNP(self,lnbg):
        return self.BetheBlochGeantNP(lnbg)
    
    def GenerateToyParticles(self, n=10000,sigma=0.01):
        n_particles = n
        
        p_ges = []
        
        for i, el in enumerate(self.masses):  
            p = np.random.uniform(0.3,10.,int(n_particles))
            mp = np.random.uniform(1/10.,1/0.3,int(n_particles))
            p_mp = 1./mp 
            p_ges.append(np.concatenate([p,p_mp]))

        signals = []
        for i, mass in enumerate(self.masses):
            ITS_tmp = []
            TPCROC0_tmp = []
            TPCROC1_tmp = []
            TPCROC2_tmp = []
            TRD_tmp = []
            TOF_tmp = []
            BBS_tmp = []
            BBA_tmp = []
            beta_tmp = []
            pmeas_tmp = []
            for p in p_ges[i]:
                bg = p/mass
                beta = bg/math.sqrt(1.+ bg*bg);
                BBS = self.BetheBlochSolidNP(math.log(bg))
                BBA = self.BetheBlochAlephNP(math.log(bg))
                ITS_tmp.append(np.random.normal(BBS,0.1*BBS) ) ## ITS dEdx = smeared gaus 10% 
                TPCROC0_tmp.append(np.random.normal(BBA,sigma*BBA) )## TPC dEdx = smeared gaus 10% for 1st layer
                TPCROC1_tmp.append(np.random.normal(BBA,sigma*BBA) )  ## TPC dEdx = smeared gaus 10% for 2nd layer
                TPCROC2_tmp.append(np.random.normal(BBA,sigma*BBA) )  ## TPC dEdx = smeared gaus 10% for 3d layer
                TRD_tmp.append(np.random.normal(BBA,0.1*BBA) )  ## TRD dEdx = smeared gaus 10% 
                TOF_tmp.append(np.random.normal(beta,0.01*beta) )  ## TOF - smeared with .... gaussian
                pmeas_tmp.append(np.random.normal(p,0.01*p))
                BBS_tmp.append(BBS)
                BBA_tmp.append(BBA)
                beta_tmp.append(beta)

            signals.append({'ITS': ITS_tmp, 'TPCROC0': TPCROC0_tmp, 'TPCROC1': TPCROC1_tmp, 'TPCROC1': TPCROC1_tmp, 
                            'TPCROC2': TPCROC2_tmp, 'TRD': TRD_tmp, 'TOF': TOF_tmp, 'BBS': BBS_tmp, 'BBA': BBA_tmp, "beta": beta_tmp,
                           "pmeas": pmeas_tmp})
        df_list=[]
        for i, val in enumerate(self.masses):
            df = pd.DataFrame.from_dict(signals[i])
            df['p'] = pd.Series(p_ges[i], index=df.index)
            df['particle'] = pd.Series(i, index=df.index)
            df_list.append(df)
        df_all = pd.concat([df_list[0],df_list[2],df_list[3],df_list[4]], ignore_index=True)
        
        return df_all
    
    
    
class DEDXEncoder():

    def __init__(self):
        self.model = None

    def AEl2_loss(y_true, y_pred):

        d = (y_true - y_pred)/ y_true 
        d0=d[:,0]
        d1=d[:,1]*10
        d2=d[:,2]
        d3=d[:,3]
        d4=d[:,4]
        d5=d[:,5]
        d=d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5

        return d/6.

    def CreateModel(self, inputdim = 7):

        def BetheBlochAleph(lnbg,kp1,kp2,kp3,kp4,kp5):# ,kp1=0.76176e-1,kp2=10.632,kp3=0.13279e-4,kp4=1.8631,kp5=1.9479):
            bg   = K.exp(lnbg)
            beta = bg/K.sqrt(1.+ bg*bg)
            aa   = K.exp(kp4*K.log(beta))
            bb   = K.exp(-kp5*K.log(bg))
            bb   = K.log(kp3+bb)
            return (kp2-aa-bb)*kp1/aa


        def BetheBlochGeant(lnbg,kp0,kp1,kp2,kp3,kp4): #kp0=2.33,kp1=0.20,kp2=3.00,kp3=173.0e-9,kp4=0.49848
            bg=K.exp(lnbg)
            mK  = 0.307075e-3
            me  = 0.511e-3
            rho = kp0
            x0  = kp1*2.303
            x1  = kp2*2.303
            mI  = kp3
            mZA = kp4
            bg2 = bg*bg
            maxT= 2*me*bg2

            x=lnbg
            lhwI=K.log(28.816e-9*K.sqrt(K.cast(rho*mZA,dtype=float))/mI)

            d2=K.switch(K.greater(x,x1),lhwI + x - 0.5,
                       K.switch(K.greater(x,x0),lhwI + x - 0.5 + (0.5 - lhwI - x0)*(((x1-x)/(x1-x0))**3),0.*bg))

            return mK*mZA*(1+bg2)/bg2*(0.5*K.log(2*me*bg2*maxT/(mI*mI)) - bg2/(1+bg2) - d2)


        def BetheBlochSolid(lnbg,kp0,kp1,kp2,kp3,kp4):
            return BetheBlochGeant(lnbg,kp0,kp1,kp2,kp3,kp4)

        def getbeta(lnbg):
            bg   = K.exp(lnbg)
            return bg/K.sqrt(1.+ bg*bg)
        
        def custom_loss(y_true, y_pred):

            d = (y_true - y_pred)/ y_true 
            d0=d[:,0]
            d1=d[:,1]*10
            d2=d[:,2]
            d3=d[:,3]
            d4=d[:,4]
            d5=d[:,5]
            d=d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5

            return d/6.

        
        inputs = Input(shape=(inputdim,))
        enc    = Dense(units=64, activation='selu')(inputs)
        enc    = Dense(units=64, activation='selu')(enc)
        enc    = Dense(units=64, activation='selu')(enc)
        #enc    = Dense(units=64, activation='selu')(enc)

        gb     = Dense(units=1, activation='linear')(enc)

        BBA_parameters = {'kp1':0.76176e-1, 'kp2':10.632, 'kp3':0.13279e-4, 'kp4':1.8631, 'kp5':1.9479}
        BBS_parameters = {'kp0':2.33,'kp1':0.20,'kp2':3.00,'kp3':173.0e-9,'kp4':0.49848}
        BBA    = Lambda(BetheBlochAleph, arguments=BBA_parameters)(gb)
        BBA4   = RepeatVector(4)(BBA)
        BBA4    = Flatten()(BBA4) 
        BBS    = Lambda(BetheBlochSolid, arguments=BBS_parameters)(gb)
        TOF    = Lambda(getbeta)(gb)
        final  = Concatenate(axis=-1)([BBS,TOF,BBA4])

        modelfi= Model(inputs=inputs,outputs=final)
        modelfi.compile(loss=custom_loss,optimizer='adam',metrics=['mse'])
        self.model= modelfi
    
    def BGfromModel(self,X):
        get_1st_layer_output = K.function([self.model.layers[0].input],
                                  [self.model.layers[4].output])
        return get_1st_layer_output(X)
