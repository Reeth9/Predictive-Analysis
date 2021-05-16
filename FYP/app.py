import streamlit as st
from PIL import Image
import csv



# 'S_per': 0.8788985632601148,
#  'C_per': 0.20583676365019996,
#  'Nodule_Count': 0.021959115975123348,
#  'Si_per': 0.27627248129577103,
#  'RT_Hum': 0.010013416053667908,
#  'Mn_per': 0.011599891186011072,
#  'Hardness_BHN': 0.014466502150139476,
#  'Mg_per': 0.003227730428852817,
#  'Nodularity_per': 0.01407539345020115



# 		C_per		Si_per		Mn_per		S_per		Mg_per		Hardness_BHN	Nodularity_per	Nodule_Count	RT_Hum
# count	658.000000	658.000000	658.000000	658.000000	658.000000	658.000000		658.000000		658.000000		658.000000
# mean	3.419416	2.508331	0.331027	0.012432	0.046997	195.787234		82.612158		284.036474		30.364742
# std		0.168605	0.163560	0.100334	0.022781	0.007226	28.163856		2.523133		47.194135		1.679160
# min		2.980000	1.890000	0.012000	0.002000	0.020000	125.000000		70.210000		145.000000		28.000000

# max		3.980000	2.898000	0.652000	0.390000	0.070000	272.000000		88.300000		451.000000		35.000000



import numpy as np 
import pandas as pd 

train = pd.read_csv("IS_project.csv")

X_tr = train[['C_per', 'Si_per', 'Mn_per', 'P_per', 'S_per', 'Cu_per', 'Mg_per',
       'Hardness_BHN', 'Nodularity_per', 'Pearlite_per', 'Ferrite_per',
       'Nodule_Count', 'Sand_AFS_Fresh', 'Sand_AFS_Returned', 'Clay_Fresh',
       'Clay_Returned', 'FRS', 'KOS', 'PH_Fresh_Sand', 'PH_Returned_Sand ',
       'Sand_temp', 'RT_Hum', 'Hum_per', 'Stripping', 'Viscosity', 'Baume ']]
y_tr = train['Remarks']


X_tr=X_tr.drop(['Clay_Returned','Ferrite_per','PH_Returned_Sand ','Pearlite_per','Stripping','Baume ','FRS','Clay_Fresh','Sand_AFS_Fresh','Sand_temp','Viscosity','PH_Fresh_Sand','Sand_AFS_Returned','Cu_per','KOS','P_per','Hum_per'],axis=1)

X_tr=X_tr.values

from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_tr, y_tr, train_size=0.9, test_size=0.1,
                                                                random_state=0)


X_train = X_train_full.copy()
X_valid = X_valid_full.copy()

import xgboost as xgb
xgbr = xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,reg_lambda=9,objective='reg:squarederror',max_depth=8)
# print(xgbr)
xtrain=X_train
ytrain=y_train

xgbr.fit(xtrain, y_train,
         early_stopping_rounds=50, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

score = xgbr.score(xtrain, ytrain)  



st.title("IH Castings Defect Prediction  ")
image = Image.open('Back_image.jpg')
newsize = (500, 360)
image = image.resize(newsize)
st.image(image)



st.sidebar.title("Parameters:")

# Carbon percentage 
# Silicon percentage 
# Manganese percentage 
# Sulphur percentage 
# Magnesium percentage 
# Hardness Brinell Hardness Number 
# Nodularity Percentage 
# Nodule Count 
# RT humidity

S_per=st.sidebar.slider("Silicon percentage:", min_value=0.002000, max_value=0.390000)

C_per=st.sidebar.slider("Carbon percentage:", min_value=2.980000, max_value=3.980000)

Nodule_Count=st.sidebar.slider("Nodule Count:", min_value=145.000000, max_value=451.000000)

Si_per=st.sidebar.slider("Sulphur percentage:", min_value=1.890000, max_value=2.898000)

RT_Hum=st.sidebar.slider("RT humidity:", min_value=28.000000, max_value=35.000000)

Mn_per=st.sidebar.slider("Manganese percentage:", min_value=0.012000, max_value=0.652000)

Hardness_BHN=st.sidebar.slider("Hardness Brinell Hardness Number:", min_value=125.000000, max_value=272.000000)

Mg_per=st.sidebar.slider("Magnesium percentage:", min_value=0.020000, max_value=0.070000)

Nodularity_per=st.sidebar.slider("Nodularity Percentage:", min_value=70.210000, max_value=88.300000)


# C_per	Si_per	Mn_per	S_per	Mg_per	Hardness_BHN	Nodularity_per	Nodule_Count	RT_Hum
# 3.488	2.425	0.271	0.008	0.047	198				81.03			305				30


# C_per	Si_per	Mn_per	S_per	Mg_per	Hardness_BHN	Nodularity_per	Nodule_Count	RT_Hum
# 3.380	2.650	0.309	0.008	0.050	190	            83.30			353				28




pred = np.array([[C_per,Si_per,Mn_per,S_per,Mg_per,Hardness_BHN,Nodularity_per,Nodule_Count,RT_Hum]])

y_RF = xgbr.predict(pred)



# st.write(y_RF[0])


if st.button('Predict'):
	if y_RF[0]==0:
		st.write("No Defect")
		st.write(f"Accuracy: {round(score*100)}%")

	else:
		st.write("Likely to have Defect")
		st.write(f"Accuracy {round(score*100)}%")


# fields=["C_per","Si_per","Mn_per","S_per","Mg_per","Hardness_BHN","Nodularity_per","Nodule_Count","RT_Hum","Remarks"]
fields=[C_per,Si_per,Mn_per,S_per,Mg_per,Hardness_BHN,Nodularity_per,Nodule_Count,RT_Hum,y_RF[0]]

if st.button('Save Data'):
	with open(r'result.csv', 'a',newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow(fields)
	    st.write("Data saved successfully")

