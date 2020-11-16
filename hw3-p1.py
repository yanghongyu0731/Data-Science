import numpy as np
import math as m

def measurement(gene,label,i):
    mu_p = mu_n = 0
    count_p = count_n = 0
    dev_p = dev_n = 0
    for k in range(62):
        if label[k] > 0:
            mu_p += gene[k,i]
            count_p += 1
        else :
            mu_n += gene[k,i]
            count_n += 1   
    mu_p /= count_p
    mu_n /= count_n
    #print(mu_p)
    #print(mu_n)    
    for k in range(62):
        if label[k] > 0:
            dev_p += (gene[k,i]-mu_p)*(gene[k,i]-mu_p)
        else :
            dev_n += (gene[k,i]-mu_n)*(gene[k,i]-mu_n)
    dev_p = m.sqrt(dev_p/count_p)
    dev_n = m.sqrt(dev_n/count_n)
    #print(dev_p)   
    #print(dev_n)   
    return abs(mu_p-mu_n)/m.sqrt(dev_p*dev_p/count_p+dev_n*dev_n/count_n)

gene = np.zeros((62,2000),float)
score = np.zeros((2000,2),float)
label = np.zeros((62),int)

f = open('Data/gene.txt','r')
for i in range(2000):
    for j in range(62):
        #print(i,' ',j)
        gene[j,i] = float(f.read(15))
        #print(gene[j,i])
    if i != 0 : f.readline()
f.closed

'''
f = open('Data/index.txt','r')
temp = ''
while temp != '\t':
    temp=f.read(1)
    print(temp,end='')
f.closed
'''

f = open('Data/label.txt','r')
for i in range(62):
    label[i] = int(f.readline())
    #print(label[i])
f.closed

for i in range(2000):
    score[i,0] = measurement(gene,label,i)
    score[i,1] = i
    #print('score = ',score[i,:])

score = score[np.lexsort(score[:,::-1].T)]

for i in range(20):
    print('score = ',score[1999-i,:])