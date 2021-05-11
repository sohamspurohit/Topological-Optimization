#!/usr/bin/env python
# coding: utf-8

# # iSURP: Topological Optimization 

# Problem: To write an iterative Python program to determine square root of 2 based on iterative calculations using a series

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integral
import sympy as sym


# ## Newton Raphson- Square root of 2

# In[ ]:


def sqroot(seed,ite):   #seed represents starting value of our iterative function, and ite is the term to which we want to iterate
    seed=0.5*(seed+2/seed) #this is the Newton-Raphson iterative series that gives us the value of square root of 2
    if ite>0:
        return sqroot(seed,ite-1)
    else:                #used a recursive function in order to get the answer
        return seed


# In[ ]:


ans=sqroot(1.0,100)   #increasing iter will give more accurate answers 
print(ans)


# ## 1-D Finite Differences Method

# In[ ]:


A=np.zeros([1000,1000])
for i in range(0,1000):                  #We first find the coefficient matrix (found using calculations), applies to u''+g=0
    for j in range(0,1000):             
        if i==j:      
            A[i][j]=2
        elif i==(j+1):
            A[i][j]=-1
        elif j==(i+1):
            A[i][j]=-1                   #Inverted it.
Ainv=np.linalg.inv(A)/(1000000)


# In[ ]:


G=np.sin(np.linspace(0,2*np.pi,1000))
U=np.dot(Ainv,G)


# In[ ]:


plt.scatter(np.linspace(0,2*np.pi,1000),U,s=0.5,c='black')
plt.grid()


# ## 2-D Finite Differences Method

# Code to implement 2-D Finite Differences Method (Refer to notes for construction of matrix)
# The expression we wish to solve is: $a\frac{\delta^2f}{\delta y^2}+b\frac{\delta^2f}{\delta x^2}+c\frac{\delta^2f}{\delta y\delta x}+g(x)=0$
# Further, for simplicity, we assume that each of $f_i,_j=0$ if i or j=0. However, there is no need to do this, and it 
# can be accounted for as boundary conditions in the array matrix of g(x,y). 
# Let n be the number of divisions of the x axis, while m is the number of divisions of m. Both are supposed to be large,
# and we assume them to be equal since we just need a good enough approximation.

# In[ ]:


print("Enter the values of the coefficients:")
a=float(input()) #We allow the user to choose the coefficients, which are real numbers
b=float(input())
c=float(input())
n=100
ctr=0
M=np.zeros((n**2,n**2))
for j in range(0,n):
    for i in range(0,n):
        M[ctr][(j)*n+i]=2*a+2*b-c
        if (i+1)<n:
            M[ctr][(j)*n+i+1]=c-b
        if (j+1)<n:
            M[ctr][(j+1)*n+i]=c-a
        if (j+1)<n and (i+1)<n:
             M[ctr][(j+1)*n+i+1]=-c
        if i>0:
             M[ctr][(j)*n+i-1]=-b
        if j>0:
             M[ctr][(j-1)*n+i]=-a
        ctr=ctr+1
Minv=np.linalg.inv(M)
#Here, we would have to make a function/ algorithm for the 2x2 matrix of g(x,y)  
for i in np.linspace(0,100,n):
    for j in np.linspace(0,100,n):         #just a skeleton
        ax[i*n+j]=f(i,j)


# ## Gradient Descent

# In[ ]:


##def func(z):                                 #just a skeleton function, returns the function whose optimum you wish to calculate
    ##return f(z)


# In[ ]:


##def der(a):                                  #function to return approximate value of the derivative
    ##return (func(a+0.001)-np.func(a))/0.001


# In[ ]:


##lim=0.3
##while 1:
    ##if der(lim)>0.001:          #iterative function with error value considered, stops when derivative reaches epsilon
        ##break
    ##lim=lim-0.002*der(lim)
##print(func(lim))


# ## Forward and Backward Euler

# In[ ]:


end_point=10
points=5001
r=2
t=np.linspace(0,end_point,points)
dt=t[1]-t[0]
N=np.zeros(points)
K=np.zeros(points)
N[0]=1
K[0]=1
for n in range (points-1):
    N[n+1]=N[n]+r*dt*N[n]
for k in range(points-1):
    K[k+1]=K[k]/(1-r*dt)
    
plt.plot(t,N,'g')
plt.plot(t, np.exp(r*t), 'r')
plt.plot(t,K,'b')
plt.legend(['Forward Euler','Exact','Backward Euler'])
plt.xlabel('t'); plt.ylabel('x(t)')


# # 1D Finite Elements Method

# Here I have implemented the 1D Finite Element Method for approximating the value of the function. Here again,
# $\frac{\delta^2u}{\delta x^2}+f(x)=0$ is solved, for x in [0,1] and $u(0)=u_0,u(1)=u_1$

# In[2]:


#Construction of the stiffness matrix
K=np.zeros([1000,1000])
for i in range(0,1000):
    for j in range(0,1000):
        if abs(i-j)>1:
            K[i][j]=0
        elif abs(i-j)==1:
            K[i][j]=-1
        elif i==j:
            K[i][j]=2
print(K)
            


# In[ ]:


values=np.linspace(0,1,1000)  #skeleton code, fn needs to be written that returns values of f(x) at x
h=1/1000
F=np.zeros(1000)
for i in range(0,1000):
    F[i]=(2/3)*(fn(values[i]-0.5*h)+fn(values[i])+fn(values[i]+0.5*h))   #Simpson's 1/3 rule used here, check notes


# In[ ]:


U=np.dot(np.linalg.inv(K),F)

