#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors,word2vec,Word2Vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors,word2vec,Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import random
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
import pandas as pd
import csv
import numpy
from sklearn import metrics 
import matplotlib.pyplot as plt
import csv
import xlrd
import numpy as np
from sklearn.model_selection import GridSearchCV


# In[2]:


tsne = TSNE(n_components=2,init='pca',early_exaggeration=12.0,learning_rate=200,n_iter=10000)
dbscan = DBSCAN(eps = 2.6,min_samples =2.8)#2.7  4    


# In[3]:


model = Word2Vec.load(r"D:\jupyter\training\models\datamodel")
# model = KeyedVectors.load_word2vec_format(r'F:\zhuomian\datamodel.txt')
df = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word2.csv',header=None)


# In[4]:


words=[]
vector= []
for i in df[0]:
        words.append(i)
for j in words:
        vector_ = list(model[j])
        vector.append(vector_)
vector = np.array(vector)
# vector = model[words]
embedd = tsne.fit_transform(vector)
dbscan.fit(embedd)
label_pred = dbscan.labels_
n_clusters = len(set(label_pred)) - (1 if -1 in label_pred else 0)


# In[5]:


# vector[373]
# model['NiMoO_4']


# In[7]:





# In[65]:


myset = set(list(label_pred))
for item in myset:
  print("the %d has found %d" %(item,list(label_pred).count(item)))


# In[66]:


df = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word2.csv',header=None)
df[3] = list(label_pred)
df.to_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word3.csv',header=None,index=None)


# In[83]:


plt.figure(figsize=(30,30))
plt.scatter(embedd[:,0],embedd[:,1])
x = {}
i = 0
label_list=[]

while i <n_clusters:
    if i == 11:
        i = i + 1
        continue
    
    x[i]=embedd[label_pred == i]
#     print(len(x[0]))
    if len(x[i])>50:
        label_1 = [i]
        label_list.append(label_1)
        plt.scatter(x[i][:,0],x[i][:,1])
    i +=1
# plt.text(embedd[1141,0],embedd[1141,1]，c = 'k')
plt.text(embedd[1140,0],embedd[1140,1],s = 'GdF_3')
plt.text(embedd[245,0],embedd[245,1],s= 'Bi_4Ti_3O_12')
plt.text(embedd[372,0],embedd[372,1],s = 'NiMoO_4')
plt.text(embedd[1119,0],embedd[1119,1],s = 'Sn-9Zn')
plt.text(embedd[2326,0],embedd[2326,1],s = 'Na_2HPO_4')
plt.text(embedd[873,0],embedd[873,1],s = 'AsGaInSb')
plt.scatter(embedd[1140,0],embedd[1140,1],c = 'k')
plt.scatter(embedd[245,0],embedd[245,1],c = 'k')
plt.scatter(embedd[372,0],embedd[372,1],c = 'k')
plt.scatter(embedd[1119,0],embedd[1119,1],c = 'k')
plt.scatter(embedd[2326,0],embedd[2326,1],c = 'k')
plt.scatter(embedd[873,0],embedd[873,1],c = 'k')
plt.show
print(label_list)


# In[52]:


# embedd


# In[70]:


# label_list = [[1], [5], [12], [27],[30],[42],[111]]
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
for k in label_list:
    try:
        df = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word3.csv',header=None)
        df2 = df[df[1].isin(k)]
        df2.to_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word4.csv',header=None,index=None)#筛选出聚类的簇放入表格
        # 创建一个无向图
        graph = nx.Graph()
        #数据准备
        model = Word2Vec.load(r"D:\jupyter\training\models\datamodel")
        df= pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word4.csv',header=None)
        name_list =list(df[0])
        edges_llist = []
        for i in name_list:
            a=1
            for j in name_list[a:]:
                    in_tuple = (i,j,model.similarity(i,j))
                    edges_llist.append(in_tuple)
            a=a+1
        # 设置有向图中的路径及权重(from, to, weight)
        graph.add_weighted_edges_from(edges_llist)
        # 计算每个节点（人）的PR值，并作为节点的pageRank属性
        pageRank = nx.pagerank(graph,max_iter=300)
        sort_d=sorted(pageRank.items(),key = lambda pageRank:pageRank[1],reverse=True)
        print(k)
        print(sort_d)
    except:
        continue


# In[45]:


embedd[1141,0],embedd[1141,1]


# In[ ]:


#Na_2HPO_4   679 21 38 57 63 68
# 1: GdF_3 Ho_2O_3  陶瓷材料
# 16 :Bi_4Ti_3O_12   薄膜材料
# 21 : NiMoO_4      电池材料
# 41: Sn-9Zn       合金
# 66 :Na_2HPO_4     溶液
# 107: AsGaInSb  Al_0.3Ga_0.7As    红外材料


# In[50]:


df = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word3.csv',header=None)
df2 = df[df[1].isin(label_list)]
df2.to_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word4.csv',header=None,index=None)#筛选出聚类的簇放入表格


# In[51]:


# for i in label_list:
#     for j in df2[df2[3]==i][0]:
#         f = open(r'F:\zhuomian\{}.txt'.format(i), 'a', encoding='UTF8')
#         print(j, end=' ', file=f)


# In[33]:


# df3 = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word2.csv',header=None)
# df4 = pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word3.csv',header=None)
# lid = df4[df4[1]==68][0]

# df3 = df3[~df3[0].isin(list(lid))]
# df3
# df3.to_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word2.csv',header=None,index=None)#筛选出不需要的簇


# In[53]:


import networkx as nx
# 创建一个有向图
graph = nx.Graph()
#数据准备
import pandas as pd
from gensim.models import Word2Vec
model = Word2Vec.load(r"D:\jupyter\training\models\datamodel")
df= pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word4.csv',header=None)
name_list =list(df[0])
edges_llist = []
for i in name_list:
    a=1
    for j in name_list[a:]:
            in_tuple = (i,j,model.similarity(i,j))
            edges_llist.append(in_tuple)
    a=a+1
# 设置有向图中的路径及权重(from, to, weight)
graph.add_weighted_edges_from(edges_llist)
# 计算每个节点（人）的PR值，并作为节点的pageRank属性
pageRank = nx.pagerank(graph)
# print(pageRank)
# # 获取每个节点的pageRank数值
# pageRank_list = {node: rank for node, rank in pageRank.items()}
# # 将paeans数值作为节点的属性
# nx.set_node_attributes(graph, name='pageRank', values=pageRank_list)
# # 画网络图
# show_graph(graph)

# #输出最大的
# for key,value in pageRank.items():
#     if(value == max(pageRank.values())):
#         print (key,value)
#输出前10
sort_d=sorted(pageRank.items(),key = lambda pageRank:pageRank[1],reverse=True)
sort_d


# In[88]:


# import pandas as pd#数据准备
# from gensim.models import Word2Vec
# model = Word2Vec.load(r"D:\jupyter\training\models\datamodel")
# df= pd.read_csv(r'F:\zhuomian\2000-2010_m2v_word_model_chem_word4.csv',header=None)
# name_list =list(df[0])
# edges_llist = []
# for i in name_list:
#     a=1
#     for j in name_list[a:]:
#             in_tuple = (i,j,model.similarity(i,j))
#             edges_llist.append(in_tuple)
#     a=a+1
# edges_llist       


# In[ ]:


7 纳米 
12 陶瓷 
21 红外探测材料 





# In[ ]:




