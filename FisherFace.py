import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
from matplotlib import pyplot as plt

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection
    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    ew, ev = linalg.eig(B, W)

    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1)
    A = A - np.tile(m, (c,1)).T
    B = np.dot(A.T, A)
    [d,v] = linalg.eig(B)

    # sort d in descending order
    order_index = np.argsort(d)
    order_index =  order_index[::-1]
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m

def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)

    return faces,idLabel

def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr


#Part 1
faces,idLabel=read_faces('train')
W,LL,m=myPCA(faces)
WE=W[:,:30]
y=np.zeros((30,120))
for i in range(0,120):
    y[:,i]=WE.T.dot(faces[:,i]-m)
z=np.zeros((30,10))
for i in range(0,10):
    z[:, i]=np.mean(y[:,12*i:12*(i+1)], 1)
mat=np.zeros((10,10))
tests,ii=read_faces('test')
cor=0
for i in range(0,120):
    mini = []
    y[:,i]=WE.T.dot(tests[:,i]-m)
    for j in range(0,10):
        mini.append(ComputeNorm(y[:,i:i+1]-z[:,j:j+1]))
    res=mini.index(min(mini))+1
    if ((i//12)+1)==res:
        cor+=1
    mat[i//12][res-1]+=1

mat=float2uint8(mat)
plt.xlabel("Output Results")
plt.ylabel("True Results")
plt.title("Accuracy={}".format(cor/120))
plt.imshow(mat,cmap='gray')
plt.savefig("PCA confusion matrix.png")
plt.show()

#Part2
mean=np.reshape(m,(160,140))
for i in range(0,8):
    fa=np.reshape(WE[:,i],(160,140))
    plt.subplot(3,3,i+1)
    plt.title("Eigenface {}".format(i+1))
    plt.axis('off')
    plt.subplots_adjust(hspace=.5)
    plt.imshow(fa,cmap='gray')
plt.subplot(3,3,9)
plt.title("Mean")
plt.axis('off')
plt.subplots_adjust(hspace=.5)
plt.imshow(mean,cmap='gray')
plt.savefig("PCA Eigenfaces.png")
plt.show()

#Part3
faces,idLabeltr=read_faces('train')
WE=W[:,:90]
yy=np.zeros((90,120))
for i in range(0,120):
    yy[:,i]=WE.T.dot(faces[:,i]-m)
LDAW, Centers, classLabels=myLDA(yy,idLabeltr)
tests,idLabelte=read_faces('test')
testfaces=np.zeros((9,120))
for i in range(0,120):
    testfaces[:,i]=LDAW.T.dot(WE.T.dot(tests[:,i]-m))
cor=0
mat=np.zeros((10,10))
for i in range(0,120):
    mini = []
    for j in range(0,10):
        mini.append(ComputeNorm(testfaces[:,i:i+1]-Centers[:,j:j+1]))
    res=classLabels[mini.index(min(mini))]
    if ((i//12)+1)==res+1:
        cor+=1
    mat[i//12][res]+=1
mat=float2uint8(mat)
plt.xlabel("Output Results")
plt.ylabel("True Results")
plt.title("Accuracy={}".format(cor/120))
plt.imshow(mat,cmap='gray')
plt.savefig("LDA confusion matrix.png")
plt.show()

#Part4
cp=LDAW.dot(Centers)
cr=WE.dot(cp)
for i in range(0,10):
    cr[:,i]+=m
    fa = np.reshape(cr[:, i], (160, 140))
    plt.subplot(2 ,5 , i + 1)
    plt.title("Center {}".format(i + 1))
    plt.axis('off')
    plt.subplots_adjust(hspace=.2)
    plt.imshow(fa, cmap='gray')
plt.savefig("LDA Centers.png")
plt.show()

#Part5
fusion=np.zeros((39,10))
fusiontest=np.zeros((39,120))
for i in range(0,10):
    for j in range(0,30):
        fusion[j,i]=z[j,i]*0.5
    for j in range(30,39):
        fusion[j,i]=Centers[j-30,i]*0.5
for i in range(0,120):
    for j in range(0,30):
        fusiontest[j,i]=y[j,i]*0.5
    for j in range(30,39):
        fusiontest[j,i]=testfaces[j-30,i]*0.5
cor=0
mat=np.zeros((10,10))
for i in range(0,120):
    mini = []
    for j in range(0,10):
        mini.append(ComputeNorm(fusiontest[:,i:i+1]-fusion[:,j:j+1]))
    res=classLabels[mini.index(min(mini))]
    if ((i//12)+1)==res+1:
        cor+=1
    mat[i//12][res]+=1
mat=float2uint8(mat)
plt.xlabel("Output Results")
plt.ylabel("True Results")
plt.title("Accuracy={}".format(cor/120))
plt.imshow(mat,cmap='gray')
plt.savefig("PCA+LDA confusion matrix.png")
plt.show()

#Part6

correctness=[]
for k in range(1,10):
    alpha=0.1*k
    fusion = np.zeros((39, 10))
    fusiontest = np.zeros((39, 120))
    for i in range(0, 10):
        for j in range(0, 30):
            fusion[j, i] = z[j, i] * alpha
        for j in range(30, 39):
            fusion[j, i] = Centers[j - 30, i] * (1-alpha)
    for i in range(0, 120):
        for j in range(0, 30):
            fusiontest[j, i] = y[j, i] * alpha
        for j in range(30, 39):
            fusiontest[j, i] = testfaces[j - 30, i] * (1-alpha)
    cor = 0
    mat = np.zeros((10, 10))
    for i in range(0, 120):
        mini = []
        for j in range(0, 10):
            mini.append(ComputeNorm(fusiontest[:, i:i + 1] - fusion[:, j:j + 1]))
        res = classLabels[mini.index(min(mini))]
        if ((i // 12) + 1) == res + 1:
            cor += 1
        mat[i // 12][res] += 1
    correctness.append(cor/120)
xx = np.linspace(0.1, 0.9, 9)
plt.title("accuracy versus alpha")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.plot(xx,correctness,marker='*', ms=10)
plt.savefig("accuracy versus alpha.png")
plt.show()


















