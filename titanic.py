import numpy as np
import pandas as pd
import random as r
import matplotlib.pyplot as plt ###importing libraries

data=pd.read_csv("/home/siddhu/Miniproject/titanic/train.csv") ###loading data
data=data.fillna(0)

datmat=np.array(data) ###convering into matrix

P=datmat[:,0] ###row matrix of passengers
P1=P.reshape(np.size(P),1) ###colomn matrix of passengers
S=datmat[:,1] ###row matrix of survival
S1=S.reshape(np.size(P),1) ###colomn matrix of survival
C=datmat[:,2] ###row marix of Class
C1=C.reshape(np.size(P),1) ###colomn marix of Class
A=datmat[:,5] ###row matrix of age
F=datmat[:,9] ###row matrix of fare
V=F.reshape(np.size(P),1) ###colomn matrix of fare
main=np.hstack((P1,C1))
main=np.hstack((main,datmat[:,5:7]))
main=np.hstack((main,V))
O=np.array([(1)])
for i in range(1,np.size(P)):
	O=np.vstack((O,np.array([(1)])))
main=np.hstack((O,main))
main1=main.reshape(6,np.size(P))
e=1e-5 
alpha=0.001


#####intializing theta##################

theta=np.array([(r.random())])
for i in range(1,6):
	theta=np.vstack((theta,np.array([(r.random())])))
theta=theta.reshape(6,1)

######geting linear function##########################

for i in range(1,10000):
	z=np.array(np.dot(main,theta),dtype=np.float128)

########geting logistic fuction#######################

	h=exp(-z)
	g=(1.0/(1.0+exp(-z)))

########Survival fuction############################

	m=np.size(z)
	J=(np.average((-S1*np.log(g+e))-((1.0-S1)*np.log(1-g+e))))

########gradient desent###########################

	grad=(1/m)*np.dot((main1),(g-S1))
	theta=theta-(alpha/m)*grad

#################final output#############################

print(J)
if J<1 and J>=0:
	print("sucess")
else:
	print('not accurate')

#################predict###################################

predict=np.array([(0)])
for i in g:
	if i<0.5:
		predict=np.vstack((predict,np.array([(0)])))
	else:
		predict=np.vstack((predict,np.array([(1)])))

f=abs(predict[1:,:]-S1)
correct=100-(((np.sum(f))/m)*100)
print(correct)









