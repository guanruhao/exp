# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


#向量加减乘除 (等长向量对应元素四则运算)
a=np.array([1,2,3])
b=np.array([0.5]*3)
print(a*b)
print(a+a)
print(a*3)
print(a/b)
x=[2,3,4,5]
y=[3,4,5,6]
B=np.array([[1,2,3],[12,3,45]])
A=np.array([[4,2,3,1],[5,4,6,8],[7,6,9,4]])
C=np.dot(B,A)
d=np.array([21,63,62])
#solve matrix Ax=d
Ainv=np.linalg.pinv(A)
x=Ainv.dot(d)

#sum a matrix along its rows/column
#row
value1=np.sum(A,1)
value1
#column
value2=np.sum(A,0)
value2

#generate matrix that normally distributed numbers
matrix1=np.random.normal(2,16,(3,5))
matrix1
#unifirm ditribution 
matrix2=np.random.uniform(-1,1,(2,4))
matrix2

#slice matrix :retain specified rows/clos
#extract
A
selection=A[np.ix_([1,2],[1,3])]
selection
#delete row/column
#row
reduce1=np.delete(A,[0,1],1)
reduce1
#column
reduce2=np.delete(A,[0,1],0)
reduce2
#检验每个值是否都满足单一条件
critical=A>2
critical
#或者
A[A>2]
A[(1<A) & (A<2)]
#实验
M=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
#先列后行（1到-2列，1到-2行
M[1:-1,1:-1]
#两个元素独立看待，（1，1），（2，2）位置的两个元素 先行后列
M[[1, 2], [1, 2]]
#求行列式
#add a new roe to matrix M denoted as N
old=np.array([[1,2,3,4],[5,2,7,2],[9,10,4,12]])
new=np.array([5,3,7,5])
print(new)
n=np.append(old,[new],0)
print(n)
det=np.linalg.det(n)
#transpose 
nt=n.transpose()
nt
dett=np.linalg.det(nt)
dett
#把向量所有元素全部变成某个值
new[:] = -1
new
#特征值和特征向量
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
results = la.eig(n)
#特征值
print(results[0])
#特征向量
print(results[1])

#常微分方程

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#求X‘=X
def diff(y, x):
	return np.array(y)
	# 上面定义的函数在odeint里面体现的就是dy/dx =y
x = np.linspace(0, 2, 101)  # 给出x范围 (0-10,100步)
y = odeint(diff, 1, x)  # 设初值为0 此时y为一个数组，元素为不同x对应的y值
# 也可以直接y = odeint(lambda y, x: x, 0, x)
print(plt.plot(x, y[:, 0]))  # y数组（矩阵）的第一列，（因为维度相同，plt.plot(x, y)效果相同）
xis1=y[round(((1-min(x))/(max(x)-min(x)))*len(x))]
x[round(((1-min(x))/(max(x)-min(x)))*len(x))]

#二阶ODE，X"=-X
def f(u,x):
  return (u[1],-x)
initial=[1,1]
x=np.linspace(0,2,101)
u=odeint(f,initial,x)
y=u[:,0]
plt.plot(x,y)


#解ODE X“=-w^2X
def f(u,x):
  return (u[1],-x)
initial=[1,1]
x=np.linspace(0,2,101)
u=odeint(f,initial,x)
y=u[:,0]
plt.plot(x,y)

asdf=1234





































