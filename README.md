## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
```
<img width="371" height="352" alt="Screenshot 2026-11-24 155750" src="https://github.com/user-attachments/assets/f95899a9-5ee1-419a-8026-10417f5ed0b1" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```
<img width="180" height="188" alt="Screenshot 2026-11-24 155839" src="https://github.com/user-attachments/assets/6a878cd0-7344-4f94-b01c-851bc6b845db" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="322" height="362" alt="Screenshot 2026-11-24 155907" src="https://github.com/user-attachments/assets/205a6ca6-c5d2-4325-9183-d7add7927bf6" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="303" height="365" alt="Screenshot 2026-11-24 160034" src="https://github.com/user-attachments/assets/b8bfcdf8-3efb-41af-8b58-6cb7d9fb8886" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="417" height="367" alt="Screenshot 2026-11-24 160114" src="https://github.com/user-attachments/assets/3c658227-d91e-4ca7-a81f-b8c5d90ef53f" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="638" height="362" alt="Screenshot 2026-11-24 160151" src="https://github.com/user-attachments/assets/6cfe3783-6e4c-489e-96df-aa2244f0e12f" />

```
pip install --upgrade category_encoders
```
<img width="840" height="303" alt="Screenshot 2026-11-24 160228" src="https://github.com/user-attachments/assets/c0b84f10-dcbb-4912-aa53-84afd3aebd80" />

```
import pandas as pd 
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```
<img width="475" height="368" alt="Screenshot 2026-11-24 160257" src="https://github.com/user-attachments/assets/d6e7b591-9932-43f7-8d31-e4cb9c088424" />

```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```
<img width="677" height="377" alt="Screenshot 2026-11-24 160324" src="https://github.com/user-attachments/assets/b180eccd-d484-4e1b-87b9-1da9ccd0ed67" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="552" height="367" alt="Screenshot 2026-11-24 160355" src="https://github.com/user-attachments/assets/fb41ff76-6569-4b7c-ab76-d33e9f72a14d" />

```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="758" height="411" alt="Screenshot 2026-11-24 160433" src="https://github.com/user-attachments/assets/c30fcae1-ff96-492a-9c81-07c3d674249e" />

```
df.skew()
```
<img width="311" height="183" alt="Screenshot 2026-11-24 160454" src="https://github.com/user-attachments/assets/da134994-1846-44f2-9f11-3b52a16ec4d9" />

```
np.log(df["Highly Positive Skew"])
```
<img width="252" height="447" alt="Screenshot 2026-11-24 160517" src="https://github.com/user-attachments/assets/c74310b4-2116-4f95-b7ab-955dd5d5e0d4" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="317" height="427" alt="Screenshot 2026-11-24 160556" src="https://github.com/user-attachments/assets/ce44befa-bc59-469b-ad63-a393a705d2a0" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="343" height="442" alt="Screenshot 2026-11-24 160619" src="https://github.com/user-attachments/assets/33e8d60c-fa8a-4099-ace0-02a984257637" />

```
np.square(df["Highly Positive Skew"])
```
<img width="366" height="413" alt="Screenshot 2026-11-24 160638" src="https://github.com/user-attachments/assets/37f0c29b-75f0-43b5-95e7-9871ad42bd40" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="822" height="440" alt="Screenshot 2026-11-24 160703" src="https://github.com/user-attachments/assets/4c2b02ec-ef8a-4c61-a506-629afa70ab40" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="365" height="251" alt="Screenshot 2026-11-24 160725" src="https://github.com/user-attachments/assets/994c8600-5281-4c7f-aad6-6a0e3b1b596e" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="811" height="457" alt="Screenshot 2026-11-24 160749" src="https://github.com/user-attachments/assets/901fb711-9d91-4e42-8b40-73061d612a62" />

```
import seaborn as sns 
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT 
plt.show()
```
<img width="627" height="452" alt="Screenshot 2026-11-24 160814" src="https://github.com/user-attachments/assets/c440a3d9-4ba8-419e-a3eb-fb3f2b9234ee" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```
<img width="627" height="445" alt="Screenshot 2026-11-24 160837" src="https://github.com/user-attachments/assets/184e704e-ef98-4577-a549-fcf2f3b9813f" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="593" height="442" alt="Screenshot 2026-11-24 160900" src="https://github.com/user-attachments/assets/669fa75e-a390-4aab-bf14-09d2d464bf08" />




# Summary :


# RESULT:
       To read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully completed
