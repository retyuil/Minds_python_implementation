import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pickle
from sklearn.neighbors import LocalOutlierFactor




def creation_liste_finale(filename,N):
	"""Main function that sanitize the data set and create a 4D point list with each of the netflows use in the data set """
	train_dt = pd.read_csv(filename)

	# checking for null values
	#print(train_dt.isnull().sum())

	# We replace Null value to Con 
	train_dt['State'] = train_dt.State.fillna(value='CON')

	# fill nan port forward 
	train_dt['Sport'] = train_dt.Sport.fillna(method = 'pad')
	train_dt['Dport'] = train_dt.Dport.fillna(method = 'pad')
	train_dt['sTos'] = train_dt.sTos.fillna(value=0.0)
	train_dt['dTos'] = train_dt.dTos.fillna(value=0.0)

	def convertlabel(sample):
	    if "Botnet" in sample: return 0
	    else: return 1
	train_dt['target'] = train_dt['Label'].apply(convertlabel)
	nb_netflow = len(train_dt['target'])


	# There are no Null values in the train data anymore
	#print(train_dt.isnull().sum().sum())


#First coordinate of the 4D point same SrcAddresse
	

	liste_point_SrcAddr = []
	for i in range(len(train_dt["SrcAddr"])):
		cpt = 0
		for j in range(1,N//2):
			if i+j > (nb_netflow - 1) or i-j < 0:
				pass
			else:
				if train_dt["SrcAddr"][i] == train_dt["SrcAddr"][i+j]:
					cpt += 1
				if train_dt["SrcAddr"][i] == train_dt["SrcAddr"][i-j]:
					cpt += 1


		liste_point_SrcAddr.append((i,cpt))


#Second coordinate of the 4D point same DstAddresse

	liste_point_DstAddr = []
	for i in range(len(train_dt["DstAddr"])):
		cpt = 0
		for j in range(1,N//2):
			if i+j > (nb_netflow - 1) or i-j < 0:
				pass
			else:
				if train_dt["DstAddr"][i] == train_dt["DstAddr"][i+j]:
					cpt += 1
				if train_dt["DstAddr"][i] == train_dt["DstAddr"][i-j]:
					cpt += 1


		liste_point_DstAddr.append((i,cpt))

#Third coordinate of the 4D point same DstAddresse and same SourcePort
	liste_point_DstAddr_Sport = []
	for i in range(len(train_dt["DstAddr"])):
		cpt = 0
		for j in range(1,N//2):
			if i+j > (nb_netflow - 1) or i-j < 0:
				pass
			else:
				if (train_dt["DstAddr"][i] == train_dt["DstAddr"][i+j]) and (train_dt["Sport"][i] == train_dt["Sport"][i+j]):
					cpt += 1
				if (train_dt["DstAddr"][i] == train_dt["DstAddr"][i-j]) and (train_dt["Sport"][i] == train_dt["Sport"][i-j]):
					cpt += 1
		liste_point_DstAddr_Sport.append(cpt)


#Fourth coordinate of the 4D point same SrcAddress and same DestPort
	liste_point_SrcAddr_Dport = []
	for i in range(len(train_dt["DstAddr"])):
		cpt = 0
		for j in range(1,N//2):
			if i+j > (nb_netflow - 1) or i-j < 0:
				pass
			else:
				if (train_dt["SrcAddr"][i] == train_dt["SrcAddr"][i+j]) and (train_dt["Dport"][i] == train_dt["Dport"][i+j]):
					cpt += 1
				if (train_dt["SrcAddr"][i] == train_dt["SrcAddr"][i-j]) and (train_dt["Dport"][i] == train_dt["Dport"][i-j]):
					cpt += 1
		liste_point_SrcAddr_Dport.append(cpt)




	liste_point_4D = []
	for i in range(len(liste_point_DstAddr)):
		liste_point_4D.append((liste_point_SrcAddr[i][1],liste_point_DstAddr[i][1],liste_point_SrcAddr_Dport[i],liste_point_DstAddr_Sport[i]))
	
	#print(liste_point_4D)
	return(train_dt,liste_point_4D)




filename =  "copie_sans_botnet.txt" #Training data set with all the botnet netflow removed
filename2 = "test.txt" #Testing data set

train_dt1,liste_point_4D = creation_liste_finale(filename,30)
train_dt,liste_point_4D2 = creation_liste_finale(filename2,30)
clf = LocalOutlierFactor(n_neighbors=5,novelty=True)
lof = clf.fit(liste_point_4D)
liste_finale = lof.predict(liste_point_4D2)

#Sanitizing the output
for i in range(len(liste_finale)):
	if liste_finale[i] == -1:
		liste_finale[i] = 0 


#Implementation of the error metrics

c_TP = 0

for i in range(len(liste_finale)):
	if liste_finale[i] == 0 and liste_finale[i] == train_dt['target'][i]:
		c_TP += 1

C_TN = 0 

for i in range(len(liste_finale)):
	if liste_finale[i] == 1 and liste_finale[i] == train_dt['target'][i]:
	 	C_TN += 1

C_FP = 0
for i in range(len(liste_finale)):
	if liste_finale[i] == 0 and liste_finale[i] != train_dt['target'][i]:
		C_FP += 1 

C_FN = 0 

for i in range(len(liste_finale)):
	if liste_finale[i] == 1 and liste_finale[i] != train_dt['target'][i]:
		C_FN += 1


FPR = C_FP/(C_TN+C_FP)
print("FPR :",FPR)

TPR = c_TP/(c_TP+C_FN)
print("TPR :",TPR)


TNR = C_TN/(C_TN+C_FP)
print("TNR :",TNR)

FNR = C_FN/(C_FN+c_TP)
print("FNR :",FNR)

precision = c_TP/(c_TP+C_FP)
print("precision :",precision)

accuracy = (c_TP+C_TN)/(c_TP+C_FP+C_TN+C_FN)
print("accuracry :",accuracy)

error_rate = (C_FP+C_FN)/(c_TP+C_FP+C_TN+C_FN)
print("error_rate :",error_rate)

f1_score= 2* ((precision*TPR)/(precision+TPR))
print("f1_score : ",f1_score)

print(c_TP,C_FN,C_TN,C_FP)