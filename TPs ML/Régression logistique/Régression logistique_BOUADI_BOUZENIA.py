#!/usr/bin/env python
# coding: utf-8

# # TP03 : Régression logistique
# 
# ## Régression linéaire polynomiale 
# Dans cette partie, comme expliqué en cours nous voyons un exemple simple d'un modèle linéaire sous forme d'un polynome multiple. 
# 
# 

# In[37]:


from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import random


# In[38]:


X = np.arange(0,4*np.pi,0.1)
Y = np.sin(X)

X = np.asarray(X)
Y = np.asarray(Y)

X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

plt.scatter(X,Y)


# In[39]:


degree = 8 #degrès du polynome résultant. 

polynomial_features = PolynomialFeatures(degree = degree)
X_TRANSF = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(X_TRANSF, Y)


# In[40]:


Y_NEW = model.predict(X_TRANSF)

rmse = np.sqrt(mean_squared_error(Y,Y_NEW))
r2 = r2_score(Y,Y_NEW)

print('RMSE: ', rmse)
print('R2: ', r2)


# In[41]:


x_new_min = 0.0
x_new_max = 10.0

X_NEW = np.linspace(x_new_min, x_new_max, 100)
X_NEW = X_NEW[:,np.newaxis]

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

Y_NEW = model.predict(X_NEW_TRANSF)

plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)

plt.grid()
plt.xlim(x_new_min,x_new_max)
plt.ylim(-1,1)

title = 'Degree = {}; RMSE = {}; R2 = {}'.format(degree, round(rmse,2), round(r2,2))

plt.title("Polynomial Linear Regression using scikit-learn and python 3 \n " + title,
          fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X,Y)
plt.show()


# ## Régression logistique 
# 
# La régression logistique est généralement utilisée à des fins de classification. Contrairement à la régression linéaire, la variable à prédire ne peut prendre qu'un nombre limité de valeurs (valeurs discrètes). 
# 
# Lorsque le nombre de résultats possibles est seulement deux, on parle de régression logistique binaire.
# 
# ![](img/logistic.JPG) 
# 
# Dans la figure ci-dessus on comprend que la régression logistique est composée d'une régression linéaire suivie de l'application d'une certaine fonction. Cette fonction est la fonction sigmoid dont voici le graphe : 
# 
# ![](img/sigmoid.JPG) 
# 

# ## 1 - Préparation des données : 
# Les données consistent en un ensemble de notes des etudiants et la valeur à prédire est si l'etudiant est admis(1) ou pas(0) 

# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[43]:


header = ["Note1", "Note2", "Admis"]
notes = pd.read_csv('marks.txt', names=header)


# In[44]:


X = notes.iloc[:, :-1].values
y = notes.iloc[:, -1].values


X.shape


# In[45]:


mean_x = X.mean()
std_x = X.std()


# In[46]:


admis = notes.loc[y == 1]
admis = admis.drop("Admis", axis=1)
non_admis = notes.loc[y == 0]
non_admis = non_admis.drop("Admis", axis=1)

plt.scatter(admis.iloc[:, 0], admis.iloc[:, 1], s=10, label='Admis')
plt.scatter(non_admis.iloc[:, 0], non_admis.iloc[:, 1], s=10, label='Non Admis')
plt.legend()
plt.show()


# ## 2- Régression logistique 
# 
# 

# In[47]:


# TODO : Calculer le sigmoid de la valeur x 
def sigmoid(x):
    # Fonction d'activation utilisée pour rendre les valeurs réelles entre 0 et 1 
    return  1 / (1 + np.exp(-x))

#Test : 
sigmoid(0)


# In[48]:


# TODO : La fonction de cout utilisée dans la régression logistique 
def J(x,y,theta): 
    t = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    cost = -(1 / t) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    gd = (1 / t) * np.dot(x.T, h - y)
    return cost, gd 


# In[49]:


# TODO : Programmer la fonction d'entrainement du modèle 
def train(x,y, theta, num_iters=400, alpha=1): 
    costs =[]
    for i in range(num_iters):
        cost, grad = J(x,y, theta)
        theta = theta - (alpha * grad)
        costs.append(cost)
    return theta , costs


# In[50]:


# TODO : fonction de normalisation des données X 
def normalisation(X):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm = (X - mean)/std
    return X_norm

# TODO : Entrainer le modèle en choisissant les bons hyperparamètres.
X  = normalisation(X)
X = np.append(np.ones((X.shape[0],1)),X,axis=1)
y=y.reshape(X.shape[0],1)
theta = np.zeros((X.shape[1],1))
print(theta.shape)
theta, couts = train(X,y, theta)
print(theta.shape)


# In[51]:


print("optimized :",theta)


# In[52]:


plt.plot(couts)


# 

# In[53]:


x_values= np.array([np.min(X[:,1]),np.max(X[:,1])])
print(theta[1].shape)
y_values = (- (theta[0] + np.dot(theta[1], x_values[0])) / theta[2],  - (theta[0] + np.dot(theta[1], x_values[1])) / theta[2])

pos , neg = (y==1).reshape(100,1) , (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="g",marker="+",label="positive")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="y",marker="*",label="negative")

plt.plot(x_values, y_values, label='separator')
plt.xlabel('axe 1 ')
plt.ylabel('axe 2 ')
plt.margins(-0.1, 0.2)
plt.legend()
plt.show()


# In[54]:


def predict(x,theta):
    z = sigmoid(np.dot(x,theta))
    preds = [0] * len(z)
    preds = np.array(preds)
    preds[z > 0.5] = 1
    return z


# ## 3- Implementation sous sklearn : 

# In[55]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

header = ["Note1", "Note2", "Admis"]
notes = pd.read_csv('marks.txt', names=header)

X = notes.iloc[:, :-1].values
y = notes.iloc[:, -1].values

# TODO : Diviser les données en données d'entrainement et données de tests (Fait dans le TP02 )
## Décider de la taille des données pour chaque set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

print(X_train.shape)
print(X_test.shape)

# Créer le modèle 
model = LogisticRegression(solver="lbfgs")

# Entraîner le modèle 
model.fit(X_train, y_train)

# Prédire les classes 
predicted_classes = model.predict(X_test)

# Calculer le score du modèle 
accuracy = accuracy_score(y_test.flatten(),predicted_classes)

print(accuracy)


# In[ ]:





# In[ ]:




