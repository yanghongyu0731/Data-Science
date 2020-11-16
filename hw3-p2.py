import pyswarms as ps
from sklearn import linear_model
import numpy as np

############################################################################################################################

label = np.zeros((62),int)
g = open('Data/label.txt','r')
for i in range(62):
    label[i] = int(g.readline())
    label[i] = label[i]/abs(label[i])
g.closed

gene = np.zeros((62,2000),float)
g = open('Data/gene.txt','r')
for i in range(2000):
    for j in range(62):
        gene[j,i] = float(g.read(15))
    if i != 0 : g.readline()
g.closed

############################################################################################################################


# Create an instance of the classifier
classifier = linear_model.LogisticRegression(max_iter=500)

# Objectibe function
def f_per_particle(m,alpha):
    total_features = 2000
    if np.count_nonzero(m) == 0 :
        gene_subset = gene
    else :
        gene_subset = gene[:,m==1]  #get the subset of the features from m   m = [0,0,1,1,......,0,1]
    classifier.fit(gene_subset,label)
    P = (classifier.predict(gene_subset) == label).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * ((gene_subset.shape[1]/total_features)))

    return j

def f(x,alpha = 0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i],alpha) for i in range(n_particles)]
    return np.array(j)

############################################################################################################################

# Initialize swarm, arbitrary
options = {'c1':1, 'c2':1, 'w':0.5, 'k':10, 'p':2}

dimensions = 2000
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions,options=options,velocity_clamp=(-5,-2))

cost, pos = optimizer.optimize(f, iters=100)


############################################################################################################################

for i in range(2000):
    if i%50 == 0:
        print("")
    print(pos[i],end=' ')

print("")
print('*'*100)

classifier.fit(gene, label)
subset_performance = (classifier.predict(gene) == label).mean()
gene_selected_features = gene[:,pos == 1]
print("origin shape = ",gene.shape)
print("subset shape = ",gene_selected_features.shape)
classifier.fit(gene_selected_features, label)
subset_performance2 = (classifier.predict(gene_selected_features) == label).mean()


print('Fullset performance: %.3f' % (subset_performance))
print('Subset performance: %.3f' % (subset_performance2))