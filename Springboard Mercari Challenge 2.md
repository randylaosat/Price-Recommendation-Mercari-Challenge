
<img src="https://www.mercari.com/assets/img/help_center/us/ogp.png"/>

# Mercari Price Suggestion Challenge
***
### Can you automatically suggest product prices to online sellers?

**Product pricing gets even harder at scale**, considering just how many products are sold online. Clothing has strong seasonal pricing trends and is heavily influenced by brand names, while electronics have fluctuating prices based on product specs.

**Mercari**, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.

In this competition, Mercari’s challenging you to **build an algorithm that automatically suggests the right product prices**. You’ll be provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

### Dataset Features

- **ID**: the id of the listing
- **Name:** the title of the listing
- **Item Condition:** the condition of the items provided by the seller
- **Category Name:** category of the listing
- **Brand Name:** brand of the listing
- **Shipping:** whether or not shipping cost was provided
- **Item Description:** the full description of the item
- **Price:** the price that the item was sold for. This is the target variable that you will predict. The unit is USD.

**Work on supply and demand**

**Source:** https://www.kaggle.com/c/mercari-price-suggestion-challenge

<img src = "https://cdn.dribbble.com/users/56196/screenshots/2281553/mobile-dribbble.gif"/>

# Representing and Mining Text
***
Since, text is the most **unstructured** form of all the available data, various types of noise are present in it and the data is not readily analyzable without any pre-processing. The entire process of cleaning and standardization of text, making it noise-free and ready for analysis is known as **text pre-processing**.

### Fundamental Concepts 

The importance of constructing mining-friendly data representations; Representation of text for data mining. 

### Important Terminologies
- **Document**: One piece of text. It could be a single sentence, a paragraph, or even a full page report. 
- **Tokens**: Also known as terms. It is simply just a word. So many tokens form a document. 
- **Corpus**: A collection of documents. 
- **Term Frequency (TF)**: Measures how often a term is in a single document
- **Inverse Document Frequency (IDF)**: distribution of a term over a corpus

### Pre-Processing Techniques
- **Stop Word Removal:** stop words are terms that have little no meaning in a given text. Think of it as the "noise" of data. Such terms include the words, "the", "a", "an", "to", and etc...
- **Bag of Words Representation: ** treats each word as a feature of the document

- **TFIDF**: a common value representation of terms. It boosts or weighs words that have low occurences. For example, if the word "play" is common, then there is little to no boost. But if the word "mercari" is rare, then it has more boosts/weight. 

- **N-grams**: Sequences of adjacent words as terms. For example, since a word by itself may have little to no value, but if you were to put two words together and analyze it as a pair, then it might add more meaning. 

- **Stemming and Lemmatization**:

- **Named Entity Extraction**: A pre-processing technique used to know  when word sequences constitute proper names. Example, "HP", "H-P", and "Hewlett-Packard" all represent the Hewlett-Packard Corporation.

- **Topic Models**: A type of model that represents a set of topics from a sequence of words. 

<img src="http://www.des1gnon.com/wp-content/uploads/2017/02/Des1gn-ON-Tendencias-no-Design-em-2017-icones-03.gif"/>

# MileStone Report 
***

**A. Define the objective in business terms:** The objective is to come up with the right pricing algorithm that can we can use as a pricing recommendation to the users. 

**B. How will your solution be used?:** Allowing the users to see a suggest price before purchasing or selling will hopefully allow more transaction within Mercari's business. 

**C. How should you frame this problem?:** This problem can be solved using a supervised learning approach, and possible some unsupervised learning methods as well for clustering analysis. 

**D. How should performance be measured?:** Since its a regression problem, the evaluation metric that should be used is RMSE (Root Mean Squared Error). But in this case for the competition, we'll be using the 

**E. Are there any other data sets that you could use?:** To get a more accurate understanding and prediction for this problem, a potential dataset that we can gather would be more about the user. Features such as user location, user gender, and time could affect it.

## General Steps

1. Handle Missing Values — Replaced “missing” values with NA.

2. Lemmatization performed on item_description — Aiming to remove inflectional endings only and to return the base or dictionary form of a word

3. Label encoding has been performed on categorical values — Encode labels with value between 0 and n_classes-1.

4. Tokenization — Given a character sequence, tokenization is the task of chopping it up into pieces, called tokens and remove punctuation.

5. Maximum length of all sequences has been specified

6. Scaling performed on target variable (price)

7. Sentiment scored computed on item_description

8. Scaling performed on item description length as well



# Import Packages
***


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
```


```python
from string import punctuation
```


```python
# vstack - adds rows, hstack - adds columns
# csr_matrix - used to handle sparse matrix
from scipy.sparse import vstack, hstack, csr_matrix
```


```python
# CountVectorizer - Simply, counts word frequencies 
# TFIDF - More importance/weights on "rare" words. Less importance/weights on "frequent" words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```


```python
# LabelBinarizer - Converts labels into numerical representation "G,B,R" -> [1,2,3]
from sklearn.preprocessing import LabelBinarizer
```


```python
# Ridge - Reduces multicollinearity in regression. Applies L2 Regularization
from sklearn.linear_model import Ridge
```

# Import Train / Test Data
***


```python
# Create training set
train = pd.read_csv('C:/Users/Randy/Desktop/training/train.tsv', sep = '\t')
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train_id</th>
      <th>name</th>
      <th>item_condition_id</th>
      <th>category_name</th>
      <th>brand_name</th>
      <th>price</th>
      <th>shipping</th>
      <th>item_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>MLB Cincinnati Reds T Shirt Size XL</td>
      <td>3</td>
      <td>Men/Tops/T-shirts</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>1</td>
      <td>No description yet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Razer BlackWidow Chroma Keyboard</td>
      <td>3</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>Razer</td>
      <td>52.0</td>
      <td>0</td>
      <td>This keyboard is in great condition and works ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>AVA-VIV Blouse</td>
      <td>1</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>Target</td>
      <td>10.0</td>
      <td>1</td>
      <td>Adorable top with a hint of lace and a key hol...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Leather Horse Statues</td>
      <td>1</td>
      <td>Home/Home Décor/Home Décor Accents</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>1</td>
      <td>New with tags. Leather horses. Retail for [rm]...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24K GOLD plated rose</td>
      <td>1</td>
      <td>Women/Jewelry/Necklaces</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>0</td>
      <td>Complete with certificate of authenticity</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create testing set
test = pd.read_csv('C:/Users/Randy/Desktop/training/test.tsv', sep = '\t',engine = 'python')
```


```python
# Create log price variable (Transformation)
y = np.log1p(train['price'])
```

# Combine Test and Train Set
***


```python
# Create combined set. You would want to apply count vectorizer on combined set so you can get the list of all possible words.
combined = pd.concat([train,test])

# Create the submission set (Only contains the test ID)
submission = test[['test_id']]

# Create size of train
train_size = len(train)
```


```python
combined.shape
```




    (1286735, 9)




```python
combined_ML = combined.sample(frac=0.1).reset_index(drop=True)
```


```python
combined_ML.shape
```




    (128674, 9)



# Part 2: Preparing the Corpus for Analysis
***

a. Remove Puncuations

b. Remove Digits

c. Remove stop words

d. Lower case words

e. Lemmatization or Stemming

##  Remove Puncuation 


```python
punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
# Create a list of punctuation replacements
punctuation_symbols = []
for symbol in punctuation:
    punctuation_symbols.append((symbol, ''))
    
punctuation_symbols
```




    [('!', ''),
     ('"', ''),
     ('#', ''),
     ('$', ''),
     ('%', ''),
     ('&', ''),
     ("'", ''),
     ('(', ''),
     (')', ''),
     ('*', ''),
     ('+', ''),
     (',', ''),
     ('-', ''),
     ('.', ''),
     ('/', ''),
     (':', ''),
     (';', ''),
     ('<', ''),
     ('=', ''),
     ('>', ''),
     ('?', ''),
     ('@', ''),
     ('[', ''),
     ('\\', ''),
     (']', ''),
     ('^', ''),
     ('_', ''),
     ('`', ''),
     ('{', ''),
     ('|', ''),
     ('}', ''),
     ('~', '')]



**Create a remove punctuation method**


```python
import string
def remove_punctuation(sentence: str) -> str:
    return sentence.translate(str.maketrans('', '', string.punctuation))
```

## Remove Digits


```python
def remove_digits(x):
    x = ''.join([i for i in x if not i.isdigit()])
    return x
```

## Remove Stop Words


```python
from nltk.corpus import stopwords

stop = stopwords.words('english')

def remove_stop_words(x):
    x = ' '.join([i for i in x.lower().split(' ') if i not in stop])
    return x
```

## Lower Case Words


```python
def to_lower(x):
    return x.lower()
```

# Part 3: Explore Training Set
***

**MIssing Values:**
- Category_name
- Brand_name
- Item_description

**Categorical Variables (Need to do Encoding):** 
- name
- category_name
- brand_name
- item_description

**Check Missing Values**


```python
train.count()
```




    train_id             593376
    name                 593376
    item_condition_id    593376
    category_name        590835
    brand_name           340359
    price                593376
    shipping             593376
    item_description     593375
    dtype: int64



**Check Data Types**


```python
train.dtypes
```




    train_id               int64
    name                  object
    item_condition_id      int64
    category_name         object
    brand_name            object
    price                float64
    shipping               int64
    item_description      object
    dtype: object



## 3a. Price Distribution
***

**Why Do Price Vary?**
- Supply and Demand
- Brand Name
- Fabric Terms
- "Quality"-Type Words (Check to see if quality plays a role in price)
- Condition 


**Summary:**
- The mean price in the dataset is **26 Dollars**
- The median price in the dataset is **17 Dollars**
- The max price in the dataset is **2000 Dollars**
- Due to the skewed dataset, the **median** price is a more reliable price to gauge off of.


```python
train.price.describe()
```




    count    593376.000000
    mean         26.689003
    std          38.340061
    min           0.000000
    25%          10.000000
    50%          17.000000
    75%          29.000000
    max        2000.000000
    Name: price, dtype: float64




```python
# Could we use these as features? Look at median price for each quantile
bins = [0, 10, 17, 29, 2001]
labels = ['q1','q2','q3','q4']
train['price_bin'] = pd.cut(train['price'], bins=bins, labels=labels)
train.groupby('price_bin')['price'].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>price_bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>q1</th>
      <td>149944.0</td>
      <td>7.710178</td>
      <td>2.083100</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>q2</th>
      <td>151863.0</td>
      <td>13.834845</td>
      <td>1.795258</td>
      <td>10.5</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>q3</th>
      <td>144043.0</td>
      <td>22.539551</td>
      <td>3.335075</td>
      <td>17.5</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>q4</th>
      <td>147215.0</td>
      <td>63.396077</td>
      <td>63.271190</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>45.0</td>
      <td>66.0</td>
      <td>2000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 7))
plt.hist(train['price'], bins=50, range=[0,250], label='price')
plt.title('Price Distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
```


![png](output_46_0.png)



```python
shipping = train[train['shipping']==1]['price']
no_shipping = train[train['shipping']==0]['price']

plt.figure(figsize=(12,7))
plt.hist(shipping, bins=50, normed=True, range=[0,250], alpha=0.7, label='Price With Shipping')
plt.hist(no_shipping, bins=50, normed=True, range=[0,250], alpha=0.7, label='Price With No Shipping')
plt.title('Price Distrubtion With/Without Shipping', fontsize=15)
plt.xlabel('Price')
plt.ylabel('Normalized Samples')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
```


![png](output_47_0.png)


## 3b. Brand Analysis
***


```python

```


```python
# Amount of unique brand names
train['brand_name'].nunique()
```




    3751




```python
# Top 20 Brand Distribution
b20 = train['brand_name'].value_counts()[0:20].reset_index().rename(columns={'index': 'brand_name', 'brand_name':'count'})
ax = sns.barplot(x="brand_name", y="count", data=b20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Top 20 Brand Distribution', fontsize=15)
plt.show()
```


![png](output_51_0.png)



```python
# Display Top 20 Expensive Brands By Mean Price
top20_brand = train.groupby('brand_name', axis=0).mean()
df_expPrice = pd.DataFrame(top20_brand.sort_values('price', ascending = False)['price'][0:20].reset_index())


ax = sns.barplot(x="brand_name", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top 20 Expensive Brand', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()# Displayd frequency ratio of brand names
train['brand_name'].value_counts(1)
```


![png](output_52_0.png)





    PINK                 0.063659
    Nike                 0.062687
    Victoria's Secret    0.056910
    LuLaRoe              0.036462
    Apple                0.020337
    Nintendo             0.017958
    FOREVER 21           0.017649
    Lululemon            0.017047
    Michael Kors         0.016415
    American Eagle       0.015604
    Rae Dunn             0.014676
    Sephora              0.014303
    Coach                0.012372
    Adidas               0.012311
    Bath & Body Works    0.012167
    Disney               0.012014
    Funko                0.011030
    Under Armour         0.009807
    Sony                 0.009587
    Old Navy             0.009208
    Hollister            0.008106
    Carter's             0.007698
    Urban Decay          0.007383
    The North Face       0.007110
    Too Faced            0.006828
    Xbox                 0.006728
    Independent          0.006711
    MAC                  0.006514
    Brandy Melville      0.006464
    Kate Spade           0.006396
                           ...   
    Sock It to Me        0.000003
    Cocomo               0.000003
    Onque Casuals        0.000003
    Kaii                 0.000003
    GoGroove Pal         0.000003
    Com                  0.000003
    Honda                0.000003
    White + Warren       0.000003
    Elomi                0.000003
    Mecca                0.000003
    Lulu Frost           0.000003
    True Rock            0.000003
    Christian Lacroix    0.000003
    Acne Jeans           0.000003
    First Act            0.000003
    Neil Allyn           0.000003
    Foundry              0.000003
    Dog MD               0.000003
    Armani Exchange      0.000003
    Oxford Golf          0.000003
    Yakima               0.000003
    Bacco Bucci          0.000003
    Bostonian            0.000003
    BedHead              0.000003
    Moose Mountain       0.000003
    Tootsie              0.000003
    Catit                0.000003
    Lisa Maree           0.000003
    Ecco Bella           0.000003
    Custo Barcelona      0.000003
    Name: brand_name, Length: 3751, dtype: float64



# 3c. Category Distribution
***


```python
def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

train['category_main'], train['category_sub1'], train['category_sub2'] = zip(*train['category_name'].apply(transform_category_name))

cat_train = train[['category_main','category_sub1','category_sub2', 'price']]

cat_train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category_main</th>
      <th>category_sub1</th>
      <th>category_sub2</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Men</td>
      <td>Tops</td>
      <td>T-shirts</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Electronics</td>
      <td>Computers &amp; Tablets</td>
      <td>Components &amp; Parts</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Women</td>
      <td>Tops &amp; Blouses</td>
      <td>Blouse</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Home</td>
      <td>Home Décor</td>
      <td>Home Décor Accents</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Women</td>
      <td>Jewelry</td>
      <td>Necklaces</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3c. Main Category

**Interesting findings:**
- Women and Beauty take up majority of the distribution
- Women and Beauty take up 56% of the distribution

**Questions to ask:**
- Can we create a gender category (Female, Male, Nuetral). Example: Three categories means three gender types. If two of them are female, then we classify as a female purchaser. If two of them are male, then we classify as male. If male/female/neutral then?
- Does gender play a role in price?
- Can we create an age category? 


```python
# Electronics have the highest std
train.groupby('category_main')['price'].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>category_main</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Beauty</th>
      <td>83315.0</td>
      <td>19.727468</td>
      <td>20.708703</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>24.0</td>
      <td>1315.0</td>
    </tr>
    <tr>
      <th>Electronics</th>
      <td>47986.0</td>
      <td>33.763889</td>
      <td>63.485958</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>30.0</td>
      <td>1909.0</td>
    </tr>
    <tr>
      <th>Handmade</th>
      <td>12257.0</td>
      <td>18.325365</td>
      <td>27.484725</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>20.0</td>
      <td>906.0</td>
    </tr>
    <tr>
      <th>Home</th>
      <td>27331.0</td>
      <td>24.845798</td>
      <td>25.203925</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>29.0</td>
      <td>848.0</td>
    </tr>
    <tr>
      <th>Kids</th>
      <td>68404.0</td>
      <td>20.664983</td>
      <td>22.877467</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>24.0</td>
      <td>809.0</td>
    </tr>
    <tr>
      <th>Men</th>
      <td>37382.0</td>
      <td>34.532369</td>
      <td>39.729618</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>40.0</td>
      <td>909.0</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>18251.0</td>
      <td>20.821434</td>
      <td>31.046225</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>1400.0</td>
    </tr>
    <tr>
      <th>Sports &amp; Outdoors</th>
      <td>9632.0</td>
      <td>25.140365</td>
      <td>27.388032</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>28.0</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>Vintage &amp; Collectibles</th>
      <td>18673.0</td>
      <td>27.158732</td>
      <td>52.338051</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>26.0</td>
      <td>1709.0</td>
    </tr>
    <tr>
      <th>Women</th>
      <td>265870.0</td>
      <td>28.843331</td>
      <td>39.435913</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>33.0</td>
      <td>2000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display distribution
train['category_main'].value_counts(1)
```




    Women                     0.451315
    Beauty                    0.141427
    Kids                      0.116116
    Electronics               0.081456
    Men                       0.063456
    Home                      0.046394
    Vintage & Collectibles    0.031697
    Other                     0.030981
    Handmade                  0.020806
    Sports & Outdoors         0.016350
    Name: category_main, dtype: float64




```python
plt.figure(figsize=(17,10))
sns.countplot(y = train['category_main'], order = train['category_main'].value_counts().index, orient = 'v')
plt.title('Top 10 Categories', fontsize = 25)
plt.ylabel('Main Category', fontsize = 20)
plt.xlabel('Number of Items in Main Category', fontsize = 20)
plt.show()
```


![png](output_58_0.png)



```python
#main = pd.DataFrame(cat_train['category_main'].value_counts()).reset_index().rename(columns={'index': 'main', 'category_main':'count'})
fig, axes = plt.subplots(figsize=(12, 7))
main = cat_train[cat_train["price"]<100]
# Use a color palette
ax = sns.boxplot( x=main["category_main"], y=main["price"], palette="Blues")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=12)

sns.plt.show()
```


![png](output_59_0.png)



```python
# Create a "no_brand" column 
train['no_brand'] = train['brand_name'].isnull()
```


```python
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y='category_main', hue='no_brand', data=train).set_title('Category Distribution With/Without Brand');
plt.show()
```


![png](output_61_0.png)


## 3c. Category_2 Distribution



```python
df = cat_train.groupby(['category_sub2'])['price'].agg(['mean']).reset_index().rename(columns={'index': 'main', 'category_main':'count'})
df= df.sort_values('mean', ascending=False).head(20)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean'], align='center', alpha=0.5, color='r')
plt.yticks(range(0,len(df)), df['category_sub2'], fontsize=15)

plt.xlabel('Price', fontsize=15)
plt.ylabel('Sub Category 2', fontsize=15)
plt.title('Top 20 2nd Category (Mean Price)', fontsize=20)
plt.show()
```


![png](output_63_0.png)


## 3c. Category_1 Distribution 


```python
df = cat_train.groupby(['category_sub1'])['price'].agg(['mean']).reset_index().rename(columns={'index': 'main', 'category_main':'count'})
df= df.sort_values('mean', ascending=False)[0:20]

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean'], align='center', alpha=0.5, color='b')
plt.yticks(range(0,len(df)), df['category_sub1'], fontsize=15)

plt.xlabel('Price', fontsize=15)
plt.ylabel('Sub Category 1', fontsize=15)
plt.title('Top 20 1st Category (Mean Price)', fontsize=20)
plt.show()
```


![png](output_65_0.png)


## 3d. Item Description Analysis

**Hypothesis:** 
- Does length play a role in price?
- Does certain descriptions make a fake item?
- Lenghthier descriptions mean more effort in item, more authentic, more valuable?


```python
# Remove Punctuation
combined.item_description = combined.item_description.astype(str)

descr = combined[['item_description', 'price']]
descr['count'] = descr['item_description'].apply(lambda x : len(str(x)))

descr['item_description'] = descr['item_description'].apply(remove_digits)
descr['item_description'] = descr['item_description'].apply(remove_punctuation)
descr['item_description'] = descr['item_description'].apply(remove_stop_words)
descr.head(3)
```

    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_description</th>
      <th>price</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>description yet</td>
      <td>10.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>keyboard great condition works like came box p...</td>
      <td>52.0</td>
      <td>188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>adorable top hint lace key hole back pale pink...</td>
      <td>10.0</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
</div>




```python
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

descr['item_description'] = descr['item_description'].apply(porter.stem)
```

    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    


```python
descr.tail(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_description</th>
      <th>price</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>693349</th>
      <td>â�—ï¸�listing â�—ï¸� brand new shades piu scar...</td>
      <td>NaN</td>
      <td>71</td>
    </tr>
    <tr>
      <th>693350</th>
      <td>rm free shipping new highly addict</td>
      <td>NaN</td>
      <td>47</td>
    </tr>
    <tr>
      <th>693351</th>
      <td>purple boys polo shirt size  old navy never worn</td>
      <td>NaN</td>
      <td>59</td>
    </tr>
    <tr>
      <th>693352</th>
      <td>express deep olive green cardigan  ultra thin ...</td>
      <td>NaN</td>
      <td>121</td>
    </tr>
    <tr>
      <th>693353</th>
      <td>shade medium neutral barley us</td>
      <td>NaN</td>
      <td>41</td>
    </tr>
    <tr>
      <th>693354</th>
      <td>flintquartz cluster self mined âœ¨measures xin...</td>
      <td>NaN</td>
      <td>243</td>
    </tr>
    <tr>
      <th>693355</th>
      <td>cosmetics travel bundle includes brow power un...</td>
      <td>NaN</td>
      <td>968</td>
    </tr>
    <tr>
      <th>693356</th>
      <td>new free shipping basstop cas</td>
      <td>NaN</td>
      <td>31</td>
    </tr>
    <tr>
      <th>693357</th>
      <td>floral kimono tropical print open front hi low...</td>
      <td>NaN</td>
      <td>94</td>
    </tr>
    <tr>
      <th>693358</th>
      <td>floral scrub tops worn less  times brown belt ti</td>
      <td>NaN</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = descr.groupby('count')['price'].mean().reset_index()
sns.regplot(x=df["count"], y=(df["price"]))
plt.xlabel("word count")
plt.show()
```


![png](output_71_0.png)


# Create Pre-Processing Functions
***


```python
combined.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Men/Tops/T-shirts</td>
      <td>3</td>
      <td>No description yet</td>
      <td>MLB Cincinnati Reds T Shirt Size XL</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Razer</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>3</td>
      <td>This keyboard is in great condition and works ...</td>
      <td>Razer BlackWidow Chroma Keyboard</td>
      <td>52.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>1</td>
      <td>Adorable top with a hint of lace and a key hol...</td>
      <td>AVA-VIV Blouse</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Home/Home Décor/Home Décor Accents</td>
      <td>1</td>
      <td>New with tags. Leather horses. Retail for [rm]...</td>
      <td>Leather Horse Statues</td>
      <td>35.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Women/Jewelry/Necklaces</td>
      <td>1</td>
      <td>Complete with certificate of authenticity</td>
      <td>24K GOLD plated rose</td>
      <td>44.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# handle_missing_values - Basic data imputation of missing values
def handle_missing_values(df):
    df['category_name'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='None', inplace=True)
    df['item_description'].fillna(value='None', inplace=True)
```


```python
# to_categorical - Converts Categorical Features 
def to_categorical(df):
    df['brand_name'] = df['brand_name'].astype('category')
    df['category_name'] = df['category_name'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')
```

# Apply Pre-Processing Functions
***


```python
# Applying the pre-processing functions
handle_missing_values(combined)
to_categorical(combined)
```


```python
# apply the pre-processing function to ML combined
handle_missing_values(combined_ML)
to_categorical(combined_ML)
```


```python
# Remove Punctuation
combined_ML.item_description = combined_ML.item_description.astype(str)

combined_ML['item_description'] = combined_ML['item_description'].apply(remove_digits)
combined_ML['item_description'] = combined_ML['item_description'].apply(remove_punctuation)
combined_ML['item_description'] = combined_ML['item_description'].apply(remove_stop_words)
combined_ML['item_description'] = combined_ML['item_description'].apply(to_lower)

combined_ML['name'] = combined_ML['name'].apply(remove_digits)
combined_ML['name'] = combined_ML['name'].apply(remove_punctuation)
combined_ML['name'] = combined_ML['name'].apply(remove_stop_words)
combined_ML['name'] = combined_ML['name'].apply(to_lower)

combined_ML.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rue21</td>
      <td>Women/Skirts/Maxi</td>
      <td>2</td>
      <td>wore big size small normally wear extra small ...</td>
      <td>maxi skirt</td>
      <td>22.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>538425.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PINK</td>
      <td>Women/Tops &amp; Blouses/Tank, Cami</td>
      <td>3</td>
      <td>pink size xs racerback free shipping</td>
      <td>vs pink dark maroon tank</td>
      <td>NaN</td>
      <td>1</td>
      <td>366252.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LuLaRoe</td>
      <td>Women/Athletic Apparel/Pants, Tights, Leggings</td>
      <td>1</td>
      <td>description yet</td>
      <td>lularoe xl irma tc leggings</td>
      <td>NaN</td>
      <td>0</td>
      <td>311147.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove Punctuation
combined.item_description = combined.item_description.astype(str)

combined['item_description'] = combined['item_description'].apply(remove_digits)
combined['item_description'] = combined['item_description'].apply(remove_punctuation)
combined['item_description'] = combined['item_description'].apply(remove_stop_words)
combined['item_description'] = combined['item_description'].apply(to_lower)

combined['name'] = combined['name'].apply(remove_digits)
combined['name'] = combined['name'].apply(remove_punctuation)
combined['name'] = combined['name'].apply(remove_stop_words)
combined['name'] = combined['name'].apply(to_lower)
```

# Create three new features from Categories (Main, Sub1, Sub2)
***


```python
combined.isnull().any()
```




    brand_name           False
    category_name        False
    item_condition_id    False
    item_description     False
    name                 False
    price                 True
    shipping             False
    test_id               True
    train_id              True
    dtype: bool




```python
combined.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>Men/Tops/T-shirts</td>
      <td>3</td>
      <td>description yet</td>
      <td>mlb cincinnati reds shirt size xl</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Razer</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>3</td>
      <td>keyboard great condition works like came box p...</td>
      <td>razer blackwidow chroma keyboard</td>
      <td>52.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>1</td>
      <td>adorable top hint lace key hole back pale pink...</td>
      <td>avaviv blouse</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>Home/Home Décor/Home Décor Accents</td>
      <td>1</td>
      <td>new tags leather horses retail rm stand foot h...</td>
      <td>leather horse statues</td>
      <td>35.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Women/Jewelry/Necklaces</td>
      <td>1</td>
      <td>complete certificate authenticity</td>
      <td>k gold plated rose</td>
      <td>44.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_ML.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rue21</td>
      <td>Women/Skirts/Maxi</td>
      <td>2</td>
      <td>wore big size small normally wear extra small ...</td>
      <td>maxi skirt</td>
      <td>22.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>538425.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PINK</td>
      <td>Women/Tops &amp; Blouses/Tank, Cami</td>
      <td>3</td>
      <td>pink size xs racerback free shipping</td>
      <td>vs pink dark maroon tank</td>
      <td>NaN</td>
      <td>1</td>
      <td>366252.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LuLaRoe</td>
      <td>Women/Athletic Apparel/Pants, Tights, Leggings</td>
      <td>1</td>
      <td>description yet</td>
      <td>lularoe xl irma tc leggings</td>
      <td>NaN</td>
      <td>0</td>
      <td>311147.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>Beauty/Hair Care/Shampoo &amp; Conditioner Sets</td>
      <td>1</td>
      <td>silk express shampoo silk conditioner leave co...</td>
      <td>silk express hair products</td>
      <td>NaN</td>
      <td>0</td>
      <td>280598.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sephora</td>
      <td>Beauty/Makeup/Face</td>
      <td>1</td>
      <td>deluxe samples ysl smashbox hourglass biossance</td>
      <td>bundle es everythings</td>
      <td>NaN</td>
      <td>1</td>
      <td>216302.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>Men/Tops/T-shirts</td>
      <td>3</td>
      <td>description yet</td>
      <td>mlb cincinnati reds shirt size xl</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Razer</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>3</td>
      <td>keyboard great condition works like came box p...</td>
      <td>razer blackwidow chroma keyboard</td>
      <td>52.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>1</td>
      <td>adorable top hint lace key hole back pale pink...</td>
      <td>avaviv blouse</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>Home/Home Décor/Home Décor Accents</td>
      <td>1</td>
      <td>new tags leather horses retail rm stand foot h...</td>
      <td>leather horse statues</td>
      <td>35.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Women/Jewelry/Necklaces</td>
      <td>1</td>
      <td>complete certificate authenticity</td>
      <td>k gold plated rose</td>
      <td>44.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



# Apply CountVectorizer / TfidfVectorizer / LabelBinarizer
***

Encode labels into categorical variables: Pandas factorize and scikit-learn LabelEncoder. 
- The result will have 1 dimension.

Encode categorical variable into dummy/indicator (binary) variables: Pandas get_dummies and scikit-learn OneHotEncoder.
- The result will have n dimensions, one by distinct value of the encoded categorical variable.

# Create new Feature (Binning Price Into Two Categories)


```python
#bins = [0, 64, 5000]
#labels = ['less','more']
#combined['lt65'] = pd.cut(combined['price'], bins=bins, labels=labels)

combined.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>category_name</th>
      <th>item_condition_id</th>
      <th>item_description</th>
      <th>name</th>
      <th>price</th>
      <th>shipping</th>
      <th>test_id</th>
      <th>train_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>Men/Tops/T-shirts</td>
      <td>3</td>
      <td>description yet</td>
      <td>mlb cincinnati reds shirt size xl</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Razer</td>
      <td>Electronics/Computers &amp; Tablets/Components &amp; P...</td>
      <td>3</td>
      <td>keyboard great condition works like came box p...</td>
      <td>razer blackwidow chroma keyboard</td>
      <td>52.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target</td>
      <td>Women/Tops &amp; Blouses/Blouse</td>
      <td>1</td>
      <td>adorable top hint lace key hole back pale pink...</td>
      <td>avaviv blouse</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>Home/Home Décor/Home Décor Accents</td>
      <td>1</td>
      <td>new tags leather horses retail rm stand foot h...</td>
      <td>leather horse statues</td>
      <td>35.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Women/Jewelry/Necklaces</td>
      <td>1</td>
      <td>complete certificate authenticity</td>
      <td>k gold plated rose</td>
      <td>44.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Apply Count Vectorizer to "name", this converts it into a sparse matrix 
cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(combined['name'])
X_name
```




    <1286735x15973 sparse matrix of type '<class 'numpy.int64'>'
    	with 4789374 stored elements in Compressed Sparse Row format>




```python
# Apply Count Vectorizer to "category_name", this converts it into a sparse matrix
cv = CountVectorizer()
X_category = cv.fit_transform(combined['category_name'])
#X_sub1 = cv.fit_transform(combined['sub_category_1'])
#X_sub2 = cv.fit_transform(combined['sub_category_2'])
X_category
```




    <1286735x1007 sparse matrix of type '<class 'numpy.int64'>'
    	with 5165431 stored elements in Compressed Sparse Row format>




```python
# Apply TFIDF to "item_description", 
tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')
X_description = tv.fit_transform(combined['item_description'])
```


```python
# Apply LabelBinarizer to "brand_name"
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(combined['brand_name'])
```

# Create CSR_Matrix & Merge the Sparse Matrices
***


```python
# Create our final sparse matrix
X_dummies = csr_matrix(pd.get_dummies(combined[['item_condition_id', 'shipping']], sparse=True).values)

# Combine everything together
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
```

# Train and Test Split
***


```python
X_train_sparse = sparse_merge[:train_size]

X_test = sparse_merge[train_size:]

#X_train = sparse_merge[:len(combined_ML)]

#X_test = sparse_merge[len(combined_ML):]
```


```python
combined.columns
```




    Index(['brand_name', 'category_name', 'item_condition_id', 'item_description',
           'name', 'price', 'shipping', 'test_id', 'train_id'],
          dtype='object')



## Cross Validation

<img src="https://cdn-images-1.medium.com/max/948/1*4G__SV580CxFj78o9yUXuQ.png"/>


```python
from sklearn.cross_validation import KFold
eval_size = .10
kf = KFold(len(y), round(1. / eval_size))
train_indicies, valid_indicies = next(iter(kf))
X_train, y_train = X_train_sparse[train_indicies], y[train_indicies]
X_valid, y_valid = X_train_sparse[valid_indicies], y[valid_indicies]
```

    C:\Users\Randy\Anaconda3\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

## Creat RMSLE Function

Since the errors are squared before they are averaged, the RMSE gives a relatively high weight to large errors. **This means the RMSE should be more useful when large errors are particularly undesirable.**

RMSE has the benefit of penalizing large errors more so can be more appropriate in some cases, for example, if being off by 10 is more than twice as bad as being off by 5. But if being off by 10 is just twice as bad as being off by 5, then MAE is more appropriate.


```python
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
```

## Train with LGBM

The reason why I used this algorithm is because it’s a good model to use on big data sets. 

It has fast:
- training sped and high efficiency
- low memory usage
- good accuracy
- good compatibility with large datasets. 

The RMSLE of LGBM is: **0.5406**


```python
import lightgbm as lgb
d_train = lgb.Dataset(X_train, label=y_train)
```


```python
params = {}
#params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'

clf = lgb.train(params, d_train, 100)
```


```python
#Prediction
lgbm_pred=clf.predict(X_valid)
```


```python
# Transform the prices back to its original price
np.expm1(lgbm_pred)
```




    array([ 10.7214129 ,  29.60000413,  13.02518987, ...,  12.85446125,
            19.31981685,  15.34901857])




```python
import time
start_time = time.time()
print('[{}] LGBM completed.'.format(time.time() - start_time))
print("LGBM rmsle: "+str(rmsle(np.expm1(y_valid), np.expm1(lgbm_pred))))
```

    [0.0] LGBM completed.
    LGBM rmsle: 0.540597319376
    

## Train with Ridge Regression

The RMSLE of Ridge Regression is: **0.4829**


```python
import time 

start_time = time.time()

model = Ridge(solver = "sag", fit_intercept=False)

print("Fitting Ridge Model")
model.fit(X_train, y_train)

preds_valid = model.predict(X_valid)

print('[{}] Ridge completed.'.format(time.time() - start_time))
print("Ridge rmsle: "+str(rmsle(np.expm1(y_valid), np.expm1(preds_valid))))
```

    Fitting Ridge Model
    [32.998536586761475] Ridge completed.
    Ridge rmsle: 0.482907420753
    


```python
np.expm1(preds_valid)
```




    array([  9.09673618,  83.84303118,  11.78868638, ...,  12.66779351,
            23.59042071,  11.21630299])



# Interesting Note

The feature **'lt65'** that I created made a significant impact on the model's performance. I binned the items into either two categories based on their price: 'Less than 65' or 'More than 65'. 

The **Ridge Regression** model's RMSLE dropped from **.4829** to **.4215** with the addition of this feature.

The **LGBM** model's RMSLE dropped from **.5406** to **.4533**

## Predict on Test Set


```python
# Predicting on never seen test set
preds = model.predict(X_test)

submission["price"] = np.expm1(preds)
submission.to_csv("submission_ridge.csv", index = False)
```

    C:\Users\Randy\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.
    


```python
submission
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_id</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>11.162749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12.555600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>53.157534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17.925542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7.363347</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>9.959583</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>9.521093</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>33.185204</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>45.666661</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>6.283195</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>52.478731</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>9.582656</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>33.508056</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>49.353728</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>24.605690</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>8.701512</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>24.601817</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>17.042603</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>41.234336</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>7.499206</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>6.479075</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>10.071822</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>11.011129</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>13.974448</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>43.893530</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>7.502994</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>20.210556</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>8.181354</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>53.241879</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>7.257089</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>693329</th>
      <td>693329</td>
      <td>28.354434</td>
    </tr>
    <tr>
      <th>693330</th>
      <td>693330</td>
      <td>20.699427</td>
    </tr>
    <tr>
      <th>693331</th>
      <td>693331</td>
      <td>14.273625</td>
    </tr>
    <tr>
      <th>693332</th>
      <td>693332</td>
      <td>43.814946</td>
    </tr>
    <tr>
      <th>693333</th>
      <td>693333</td>
      <td>16.261933</td>
    </tr>
    <tr>
      <th>693334</th>
      <td>693334</td>
      <td>81.738151</td>
    </tr>
    <tr>
      <th>693335</th>
      <td>693335</td>
      <td>6.648241</td>
    </tr>
    <tr>
      <th>693336</th>
      <td>693336</td>
      <td>15.763434</td>
    </tr>
    <tr>
      <th>693337</th>
      <td>693337</td>
      <td>17.041259</td>
    </tr>
    <tr>
      <th>693338</th>
      <td>693338</td>
      <td>14.168938</td>
    </tr>
    <tr>
      <th>693339</th>
      <td>693339</td>
      <td>13.545883</td>
    </tr>
    <tr>
      <th>693340</th>
      <td>693340</td>
      <td>3.973624</td>
    </tr>
    <tr>
      <th>693341</th>
      <td>693341</td>
      <td>219.385862</td>
    </tr>
    <tr>
      <th>693342</th>
      <td>693342</td>
      <td>4.910538</td>
    </tr>
    <tr>
      <th>693343</th>
      <td>693343</td>
      <td>17.301045</td>
    </tr>
    <tr>
      <th>693344</th>
      <td>693344</td>
      <td>21.939166</td>
    </tr>
    <tr>
      <th>693345</th>
      <td>693345</td>
      <td>13.895224</td>
    </tr>
    <tr>
      <th>693346</th>
      <td>693346</td>
      <td>29.366304</td>
    </tr>
    <tr>
      <th>693347</th>
      <td>693347</td>
      <td>45.978350</td>
    </tr>
    <tr>
      <th>693348</th>
      <td>693348</td>
      <td>67.798859</td>
    </tr>
    <tr>
      <th>693349</th>
      <td>693349</td>
      <td>10.384145</td>
    </tr>
    <tr>
      <th>693350</th>
      <td>693350</td>
      <td>8.716605</td>
    </tr>
    <tr>
      <th>693351</th>
      <td>693351</td>
      <td>9.640998</td>
    </tr>
    <tr>
      <th>693352</th>
      <td>693352</td>
      <td>14.098605</td>
    </tr>
    <tr>
      <th>693353</th>
      <td>693353</td>
      <td>14.731720</td>
    </tr>
    <tr>
      <th>693354</th>
      <td>693354</td>
      <td>19.804941</td>
    </tr>
    <tr>
      <th>693355</th>
      <td>693355</td>
      <td>26.429490</td>
    </tr>
    <tr>
      <th>693356</th>
      <td>693356</td>
      <td>6.252158</td>
    </tr>
    <tr>
      <th>693357</th>
      <td>693357</td>
      <td>15.377224</td>
    </tr>
    <tr>
      <th>693358</th>
      <td>693358</td>
      <td>9.742211</td>
    </tr>
  </tbody>
</table>
<p>693359 rows × 2 columns</p>
</div>



# LeaderBoard Result (Top 36%)

<img src = "http://i63.tinypic.com/14ccuv6.jpg" /img>

# Conclusion 
***

I am happy to have done this competition because it has opened up my mind into the realm of NLP and it showed me how much pre-processing steps are involved for text data. I learned the most common steps for text pre-processing and this allowed me to prepare myself for future work whenever I’m against text data again. Another concept that I really learned to value more is the choice of algorithms and how important computation is whenever you’re dealing with large datasets. It took me a couple of minutes to even perform some data visualizations and modeling. Text data is everywhere and it can get messy. Understanding the fundamentals on how to tackle these problems will definitely help me out in the future.


```python

```
