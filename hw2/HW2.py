
# coding: utf-8

# In[1]:


print('My name is Zhicheng (Jason) Xue')


# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
random.seed(2018)


# ### 1.  Placeholder for an input array with dtype float32 and shape None
# ### 2.  Scopes for the input, middle section and final node f

# In[3]:


graph = tf.Graph()


# In[4]:


with graph.as_default():
    
    #Scope for the input section 
    with tf.name_scope(name='Input_placeholder'):
        #1.  Placeholder for an input array with dtype float32 and shape None
        a = tf.placeholder(shape=None, dtype=tf.float32, name='input_a')
        
    #Scope for the middle section 
    with tf.name_scope(name='Middle_section'):
        
        b = tf.reduce_prod(input_tensor=a, name='product_b')
        c = tf.reduce_mean(input_tensor=a, name='mean_c')
        d = tf.reduce_sum(input_tensor=a, name='sum_d')
        e = tf.add(b,c, name='add_e')
        
    #Scope for the final node   
    with tf.name_scope(name='Final_node'):
        
        f = tf.multiply(x=e,y=d,name='mul_f')


# ### 3.  Feed the placeholder with an array A consisting of 100 normally distributed random numbers with Mean = 1 and Standard devia=on = 2

# In[5]:


input_array=np.random.normal(1,2,100)


# In[6]:


print(input_array)


# In[7]:


replace_dict={a:input_array}


# In[8]:


sess=tf.Session(graph=graph)


# In[9]:


sess.run(a,feed_dict=replace_dict)


# In[10]:


sess.run(b,feed_dict=replace_dict)


# In[11]:


sess.run(c,feed_dict=replace_dict)


# In[12]:


sess.run(d,feed_dict=replace_dict)


# In[13]:


sess.run(e,feed_dict=replace_dict)


# In[14]:


sess.run(f,feed_dict=replace_dict)


# ### 4.  Save your graph and show it in TensorBoard

# In[16]:


writer=tf.summary.FileWriter('./hw2',graph=graph)


# ![TensorBoard](./hw2/HW2_TensorBoard.png)

# In[17]:


writer.close()


# In[18]:


sess.close()


# ### 5.  Plot you input array on a separate figure

# In[19]:


# histogram of the input array
n, bins, patches = plt.hist(input_array, 5, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Random Normal Number')
plt.ylabel('Probability')
plt.title('Histogram of Random Generated Array From Normal(1,2)',)
plt.text(3, .2, r'$\mu=1,\ \sigma=2$')
plt.axis([-5, 7, 0, 0.25])
plt.grid(True)
plt.show()

