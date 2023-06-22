import pandas as pd
import pickle
import numpy as np
model=pickle.load(open("/Users/vn55le3/Downloads/DS2023Talendig/7-LogisticRegression/modelo_logistico.sav",'rb'))
X=[]
for i in range(4):
    cad="Digite la caracteristica"+str(i+1)
    d=float(input(cad))
    X.append(d)
X=np.array(X)
pred=model.predict(X.reshape(1,-1))
print("La clase predicha es "+str(pred))

