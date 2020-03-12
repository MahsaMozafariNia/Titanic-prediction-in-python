# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:41:57 2020
Titanic Porject
Mahsa mozafariNia
It is completed.
@author: 2016
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import re
import seaborn as sns

data=pd.read_csv("D:/Old_Data/math/Data science toseeh/My data sets for testing code/Titanic-kaggle/train.csv",na_values=' ')
data.columns
data.index
data.info()
kholase=data.head(100)
aa=data.groupby("Survived").agg("mean")
bb=data.groupby("Survived").agg("std")

grid=sns.FacetGrid(hue="Survived",data=data)
grid.map(plt.hist,"Age",alpha=0.4,bins=20)
plt.legend()
'''
در ستون نام اسم و فامیل افراد به همراه یک پسوند اقا، خانم، دوشیزه... آمده است.
 باید ستونی ایجاد کنیم که شامل این پسوند ها باشه 
و ستون اسامی را حذف کنیم. چون اسم افراد در پیش بینی تاثیری ندارد 
و ممکن است باعث اشتباه شود مثلا اگر افرادی با 
نام علی فوت کرده بودند هر داده تست با نام علی را ممکن است فوت شده لیبل گذاری کند.

بعد از این پسوند ها یک دات آمده است پس برای استخراج آنها باید از دستور زیر استفاده کنیم.

'''
data["Title"]=data.Name.str.extract("([A-Za-z]+)\.",expand=False)

'''
یا میتوان دستور زیر را اجرا کرد
Feature['Salutation'] = Feature['Name'].str.split(",").str[1].str.split().str[0]
'''

'''
توضیخات در لینک زیر:
https://docs.python.org/3/library/re.html
    
'''
np.unique(data["Title"])

'''
'Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady',
'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev','Sir'
احتمالا اولی کاپیتان است و دومی سرهنگ
این رده ها اهمیت دارد چون ممکن است خانم ها و دوشیزه ها و بچه ها را نجات داده باشند
 یا مثلا کاپیتان راه کارهاییی برای نجات خودش داشته باشد...

با جستجو در نت متوجه شدم که یک سری از چسوند ها در حالت فرانسوی بیان شده مانند
Mlle=Miss 
Mme=madam=Mrs
 چون ممکنه در نجات دادن کشور های خاصی را نجات داده باشند و ستون کشور مهم باشه مسخواستم ستونی با نام کشور ایجاد کنم
اما وقتی  اسم افراد را در نت سرچ کردم دیدم که ملیت های متفاوتی وجود داره 
و پسوند برای ملیت دیگر است.
پس نمیتوان ستونی به نام کشور ایجاد کرد
don=sir=mr
mrs=ms=lady

بچه ها چه پسوندی دارند؟
بعدا خواهیم دید
'''

data["Title"].value_counts()

'''
Mr-517      Miss-182      Mrs-125      Master-40     Dr-7         Rev-6
Major-2     Mlle-2        Col-2        Ms-1          Countess-1
Mme-1       Sir-1         Lady-1       Capt-1        Don-1        Jonkheer-1

'''



'''
اگر بخواهیم ببینیم از هر کدام در زن ها و مرد ها چند تا امده:
    
'''

data.groupby("Title")["Sex"].value_counts()

'''
یا اینکه به صورت زیر بنویسیم که مرتب تر است

'''

pd.crosstab(data["Title"],data["Sex"])




'''
رده سنی هر یک از پسوند ها

'''
rade_seni=data.groupby("Title")["Age"].mean()

'''
capt-70(یک مرد)  col-58 (دو مرد)   countness-33 (فقط یک خانم بود)
Don-40   Dr-42  Jonkheer-38  Lady-48   Major-48.5   Master-4.57
Miss-21.77   Mlle-24     Mme-24   Mr-32   Mrs-35
Ms-28       Rev-43         Sir-49

'''
 

'''
میخواهم پسوند های زیر را داشته باشم
don=sir=mr
Mrs=ms=lady=countess(is a royla name for Queens who are married.)
Mlle=Miss (in itallian we have Mlle (wikipedia))
Mme=madam in itally and is equal to Mrs in english
Jonkheer=Mr
پسنوند فرزاندان اشفرافیت یا شوالیه ها است. البته این فرد در داده دارای 38 سال سن است.
Rev=Mr
Major=Mr
Dr
colone
captain
Master

البته یک بار هم همین روند را انجام میدهم اما کاپیتان و سرهنگ را همان اقا در نظر میگیرم چون تعداداین افراد در اموزشی کم است.
'''
data.loc[data["Title"]=="Sir","Title"]="Mr"
data["Title"].replace("Don","Mr",inplace=True)
data["Title"].replace("Ms","Mrs",inplace=True)
#data.loc[data["Title"]=="Ms" ,"Title"]="Mrs"
data["Title"].replace("Lady","Mrs",inplace=True)
data["Title"].replace("Mme","Mrs",inplace=True)
data["Title"].replace("Mlle","Miss",inplace=True)
data["Title"].replace("Countess","Mrs",inplace=True)
data["Title"].replace("Jonkheer","Mr",inplace=True)
data["Title"].replace("Rev","Mr",inplace=True)
data["Title"].replace("Major","Mr",inplace=True)

#2 khate zir marboot be colonel va capitan va Dr ra yek bar be mr
#tabdil kardam va bare digar na va ejra kardam.
data["Title"].replace("Capt","Mr",inplace=True)
data["Title"].replace("Col","Mr",inplace=True)
data["Title"].replace("Dr","Mr",inplace=True)
data.drop("Name",axis=1,inplace=True)
np.unique(data["Title"])

'''
حال میخواهیم ببینیم از هر کدام از این پسوند ها چه درصدی زنده مانده اند.
چون داده ها صفر و یک هستند پس درصد زنده ها میشه همان میانگین

'''
tedad_zandeha=data.groupby("Title")["Survived"].mean()
tedad_zandeha

'''
میبینیم که به ترتیب خانم ها، دو شیزه ها، جوانان با
میانگین سنی 24 سال بیشترین درصد زنده ها را داشته اند پس احتمالا در اولویت بودند برای قایق نجات
به نظرم دکتر و سرهنگ...را نمیشه حرفی زد چون در کل دو تا سرهنگ داشتیم و یکی زنده مانده
 و یکی مرده پس نمیشه گفت در اولویت بودن برای سوار شدن به قایق نجات. 

'''

'''
بعضی اطلاعات که جمع میکنیم فقط به خاطر است که ببینیم ایا داده واقعی به نطر میرسد و با عقل جور در میاد یا نه

'''
pcalss_survived=data.groupby("Pclass")["Survived"].mean()
pcalss_survived

'''
نتیجه:
 class 1: 0.62    class 2: 0.47   class 3: 0.24
 .منطقی به نطر میرسد افرادی که در اتاق ها بهتر بوده اند بنابر ساختار کشتی
 در سمت بالای کشتی بوده اند و وقتی اب کشتی را گرفته 
 اتاق های مربوط به کلاس 3 اول غرق شده و پر از اب شده
 همچنین شاید به دلیل پولی زیادی که کلاس یک و 2 پرداخت کرده اند
 احتمالا مربوط به اشراف زاده ها بوده آن ها اولویت برای سوار به قایق نجات داشته اند.

'''

sib_surv=data.groupby("Survived")["SibSp"]
plt.figure()
plt.hist(sib_surv.get_group(0),facecolor="Red",alpha=0.3)
plt.hist(sib_surv.get_group(1),facecolor="blue",alpha=0.3)
plt.xlabel="SibSp"
plt.ylabel="frequency"
plt.title="Histogram"

'''5
راه های دیگه
grid=sns.FacetGrid(data,col="Survived")
grid.map(plt.hist,"SibSp",bins=40)

sns.countplot(data["SibSp"],hue=data["Survived"])
'''
corrolation=data.corr()
sns.heatmap(corrolation,vmin=0,vmax=1,cmap="summer")

sns.pairplot(data,hue="Survived")
'''
به نظرم اطلاعات خاصی نداد.
'''

'''
The FacetGrid class is useful when you want to visualize the distribution of a 
variable or the relationship between multiple variables separately within subsets
of your dataset. A FacetGrid can be drawn with up to three dimensions:
row, col, and hue. The first two have obvious correspondence with the resulting
array of axes; think of the hue variable as a third dimension along a depth axis,
where different levels are plotted with different colors.
'''

'''
تصویر سازی نمودار ها

'''
'''
تعداد مرده ها و زنده ها

'''
plt.hist(df["Survived"],50,facecolor="b",edgecolor="g")

'''
تعداد زن ها و مردانی که زنده مانده اند

'''
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
age_Serv=sns.FacetGrid(data,col="Survived")
age_Serv.map(plt.hist,"Age",bins=40)

'''
با توجه به این نمودار افراد مسن و بچه ها تقریبا زنده مانده اند.
پس میشه گفت عامل سن در مردن یا زنده بودن تقریبا موثر بوده.

'''

data["Family"]=data["Parch"]+data["SibSp"]
family_surv=sns.FacetGrid(data,col="Survived",hue="Pclass",palette="Set1")
family_surv.map(plt.hist,"Family",alpha=0.5,bins=20)
family_surv.add_legend()
'''
میبینیم که اکثر افرادی که مرده اند هیچ فامیلی نداشته اند 
و تنها بوده اند اما در مورد بقیه حرف خاصی نمیتوان زد.
اکثر افرادی که فوت کرده اند مربوط به کلاس 3 هستند. 
همچنین میبینیم که با اینکه تعداد اعضای خانداده 4 تا 8 هست برخلاف تصور 
ما بیشتر این افراد فوت کرده اند. دلیل ان این است که از کلاس 3 بوده اند.
اما چون فقط از صفر بودن یا نبودن خانواده اطلاعات کسب کردیم پس بهتر است ستون تنها و با کسی بودن را اضافه کنیم.
'''
def alone(x):
    if x==0:
        x=0
    else:
        x=1
    return(x)
        
data["Alone"]=data["Family"].agg(alone)
#0 means he is alone and 1 means he is with some one.
data.drop("Parch",axis=1,inplace=True)
data.drop("SibSp",axis=1,inplace=True)


class_surv=sns.FacetGrid(data,col="Survived")
class_surv.map(plt.hist,"Pclass")

'''
اکثرا افراد فوت شده در کلاس 3 هستند.

'''
'''
عنوان ادمها چون اسمی است پس به جای هیستوگرام باید بار پلات رسم کرد
پیشرفنه تر از بار پلات همان کانت پلات است.
'''
sns.countplot(x="Title",data=data,hue="Survived")

'''
میتوان دید که اکثر اقایان فوت کرده اند و خانم ها بچه ها 
و دوشیزه ها به نسبت خودشان زندده مانده اند. توجه شود چون تعداد 
اقایان و خانم ها برابر نیست نمیتوان با هم مقایسه کرد.

'''
sns.countplot(x="Embarked",data=data,hue="Survived")

'''
حال به داده های گم شده میپردازیم.

'''
'''
میخواهیم داده های گمشده سن را بر اساس عنوان اسم انها مقدار دهی کنیم یعنی گرون مستر 
که میانگین سنی کمی داشتند اگر سن گمشده داشتند با ان میانگین پر کنیم.
از
groupby-transform
استفاده میکنیم.
'''

np.sum(data.isna())
data["Age"].fillna(value=data.groupby("Title")["Age"].transform("mean"),inplace=True)
data["Embarked"].value_counts()
# max is for C. pas missing ha ra ba c por mikonim.
data["Embarked"].fillna(value="C",inplace=True)
data["Cabin"].value_counts()
# max is for "C23 C25 C27". pas missing ha ra ba "C23 C25 C27" por mikonim.
data["Cabin"].fillna(value="C23 C25 C27",inplace=True)

'''
حال میخواهیم پیش بینی انجام دهیم. قبل از ان باید لیبل تمام داده های کیفی را تغییر دهیم
دو را داریم یکی الگوریتم 
LableEncoder
که کافی است روی ستون مورد نظر فیت کرده و سپس انتقال انجام داده.
راه دیگر با اصتفاده از نگاشت مپ و اصتفاده از دیکشنری است که در صورتی که لیبل ها زیاد باشد فایده نداره. 
'''

data.dtypes
data["Embarked"]=data["Embarked"].map({"S":1,"C":2,"Q":3})

encoder=preprocessing.LabelEncoder()

encoder.fit(data["Title"])
data["Title"]=encoder.transform(data["Title"])

encoder.fit(data["Sex"])
data["Sex"]=encoder.transform(data["Sex"])

encoder.fit(data["Ticket"])
data["Ticket"]=encoder.transform(data["Ticket"])

encoder.fit(data["Cabin"])
data["Cabin"]=encoder.transform(data["Cabin"])

'''
lable encoder 
برای متغیر های کیفی که دارای ترتیب هستند استفاده میشه 
get_dummes
 برای کیفی های غیر ترتیبی و یک پارامتر که که اگر 10 کلاس داریم باید برابر با 9 بذاریم که یکی بشه مبنا.

'''
یا میتوان
LabelCol = [1,5,7]
for i in LabelCol:
    LabelX = LabelEncoder()
    X[:,i]=LabelX.fit_transform(X[:,i])
    که مثلا 1و5و7 ستون هایی هستند که باید لیبل بندی بشن.
    
'''


'''
df=data.drop(labels="Cabin",axis=1,inplace=False)
df.columns
df.index
#df.dropna(axis=0,inplace=True)
np.sum(df.isna())
'''

'''
df.describe(include="all")
df.dtypes

'''

'''
dfsurvivd=df.groupby(["Survived","Sex"])
plt.hist(dfsurvived.value_counts())
'''
'''
Hal mikhahim regression anjam dahim. chon pasokh 
0 ya 1 ast pas bayad logistic regression anjam dahim.

'''

y=data["Survived"]
X=data.drop(labels="Survived",axis=1)

 '''
    algorithm haye mokhtalef ra baraye pishbini anjam midahim bebinim kodoom behtare
    
    mesle logistic regresssion,LDA,QDA,NBayes,Knn (ke non parametric ast)
    
    '''
lr=LogisticRegression(penalty="l2",solver="newton-cg")
LDA=LinearDiscriminantAnalysis()
QDA=QuadraticDiscriminantAnalysis()
gnb=GaussianNB()
L1=[];L2=[]
LD1=[];LD2=[]
QD1=[];QD2=[]
GN1=[];GN2=[]
for i in range(1,10000):
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    
    lr.fit(X_train,y_train)
    y_predict=lr.predict(X_test)
    Lmatrix=confusion_matrix(y_test,lr.predict(X_test))
    L1.append(Lmatrix[0][0]/(Lmatrix[0][0]+Lmatrix[0][1]))
    L2.append(Lmatrix[1][1]/(Lmatrix[1][0]+Lmatrix[1][1]))
    
    LDA.fit(X_train,y_train)
    y_predict_lda=LDA.predict(X_test)
    LDAmatrix=confusion_matrix(y_test,LDA.predict(X_test))
    LD1.append(LDAmatrix[0][0]/(LDAmatrix[0][0]+LDAmatrix[0][1]))
    LD2.append(LDAmatrix[1][1]/(LDAmatrix[1][0]+LDAmatrix[1][1]))
    
    QDA.fit(X_train,y_train)
    y_predict_qda=QDA.predict(X_test)
    QDAmatrix=confusion_matrix(y_test,QDA.predict(X_test))
    QD1.append(QDAmatrix[0][0]/(QDAmatrix[0][0]+QDAmatrix[0][1]))
    QD2.append(QDAmatrix[1][1]/(QDAmatrix[1][0]+QDAmatrix[1][1]))
      
    gnb.fit(X_train,y_train)
    y_predict_gnb=gnb.predict(X_test)
    gnbmatrix=confusion_matrix(y_test,gnb.predict(X_test))
    GN1.append(gnbmatrix[0][0]/(gnbmatrix[0][0]+gnbmatrix[0][1]))
    GN2.append(gnbmatrix[1][1]/(gnbmatrix[1][0]+gnbmatrix[1][1]))
    
'''
miyangin darsade dorostiye kelase aval
dar har yek az algorithm ha bad az 1000 bar ejra ba test tasadofi

'''
print(np.mean(L1),np.mean(LD1),np.mean(QD1),np.mean(GN1))

'''
miyangin darsade dorostiye kelase dovom
dar har yek az algorithm ha bad az 1000 bar ejra ba test tasadofi

'''

print(np.mean(L2),np.mean(LD2),np.mean(QD2),np.mean(GN2))
#print('Logistc Regression:\n\n',confusion_matrix(y_test,lr.predict(X_test)),'\n')
#print("gnb\n\n",confusion_matrix(y_test,y_predict_gnb),"\n")
#print("QDA\n\n",confusion_matrix(y_test,y_predict_qda),"\n")
#print("LDA\n\n",confusion_matrix(y_test,y_predict_lda),"\n")
  
#lr.score(X_test,y_test)
#gnb.score(X_test,y_test)
#LDA.score(X_test,y_test)
#QDA.score(X_test,y_test)
'''
میتوان موارد زیر را نیر برای هر یک از الگوریتم ها بدست اود
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
recall_score(Y_Test, y_pred)     
 tp / (tp + fn)
precision_score(Y_Test, y_pred)
it is tp/(tp+fp)
f1_score(Y_Test, y_pred)
F1 = 2 * (precision * recall) / (precision + recall)


print(classification_report(y_test,y_pred))
همه ویژگی های بالا را در یک جدول با هم نشان میده.

'''




'''
Useful links:
    https://www.kaggle.com/sagaramu/titanic-visualization-with-additional-features

'''