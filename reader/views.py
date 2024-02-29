from django.shortcuts import render
from django.conf import settings
import pandas as pd
from django.http import HttpResponse
from django.http import JsonResponse
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud    
from nltk import tokenize
import nltk
import seaborn 
import string
from sklearn import metrics
import itertools
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import os
from datetime import datetime

plt.switch_backend('agg')

filename="news.csv"
def reuploadfile(request): 
    context={} 
    df= pd.read_csv("news-sample.csv")  
     
    context = { 
        "head": [], 
    }   
    for v in df.head().to_dict('records'):
       context['head'].append(v.items())   
     
    return render(request, 'reuploadfile.html', context=context)
    
def savefile(request): 
   try:
    file=request.FILES['file1']  
    output_file = open(filename, "wb")           
    output_file.write(file.read()) 
    output_file.close()
    dt_epoch = datetime.now().timestamp() 
    os.utime(filename, (dt_epoch, dt_epoch))   
    return JsonResponse({"success":1,"message":"File Uploaded Successfully."});     
   except Exception as e:
    return JsonResponse({"success":0,"message":str(e)});   

# Create your views here.
def home(request):
  try:
    page = request.GET.get('page', 1)
    search = request.GET.get('search', "") 
    df= pd.read_csv(filename) 
    context = {
        "success": True,
        "data": [],
        "search": search
    }
    if search:
           df["Indexes"]= df["text"].str.find(search) 
           df = df[df["Indexes"] != -1]
           context['rcount']=len(df)
           

    # seprating the necessary data 
    titles=list(df.title)
    descs=list(df.text)
    labels=list(df.label)
    for i in range(0,5): 
        context["data"].append({
            "title": titles[i],
            "description": descs[i][0:100]+"...",
            "label":labels[i].lower()
        })
    # send the news feed to template in context
    return render(request, 'index.html', context=context)
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})        
    
def loadcontent(request):
    try:
        page = int(request.GET.get('page', 1))
        search = request.GET.get('search', None)
        df= pd.read_csv(filename) 
        if search:
           df["Indexes"]= df["text"].str.find(search) 
           df = df[df["Indexes"] != -1]
           
        context = {
        "success": True,
        "data": [],
        "search": search
        }
    # seprating the necessary data 
        titles=list(df.title)
        descs=list(df.text)
        labels=list(df.label)
        for i in range((page-1)*5,((page-1)*5)+5): 
           context["data"].append({
            "title": titles[i],
            "description": descs[i][0:100]+"...",
             "label":labels[i].lower()
          })

        return JsonResponse(context)
    except Exception as e:
        return JsonResponse({"success":False,"message":str(e)})  
        
 

def explorecount(request):
  try:
    df= pd.read_csv(filename) 
    data = df 
    dcount=data.groupby(['label'])['label'].agg(list)
    datacount={}
    for i in dcount:
        datacount[i[0]]=len(i)
    data.groupby(['label'])['label'].count().plot(kind="bar") 
    plt.savefig(settings.STATIC_ROOT+'\\visualize.png')  
    plt.close()
    
    context={'datacount':datacount.items(),'visualize':'visualize.png'}
    
    return render(request, 'explorecount.html', context=context)  
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})
    
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
 
def cleandata(data):
     data['text'] = data['text'].apply(punctuation_removal) 
     stop = open("stopwords", 'r').read()
     stop = stop.splitlines()
     data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
     return  data 
    
    
def explorewordcloud(request):
  try:
    df= pd.read_csv(filename)  
    data=cleandata(df)    
    dcount=data.groupby(['label'])['label'].agg(list)
    datacount={}
    for i in dcount:
        datacount[i[0]]= len(i)    
        fake_data = data[data["label"] == i[0]]
        all_words = ' '.join([text for text in fake_data.text])
        wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")    
        plt.savefig(settings.STATIC_ROOT+'\\'+i[0]+'-wordcloud.png')  
        plt.close()
        
    context={'datacount':datacount.items()}
    
    return render(request, 'explorewordcloud.html', context=context)       
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})
    
def explorewordcount(request):
  try:
    df= pd.read_csv(filename) 
    data=cleandata(df)    
    dcount=data.groupby(['label'])['label'].agg(list)
    datacount={}
    for i in dcount:
        datacount[i[0]]={'total':len(i)}  
        token_space = tokenize.WhitespaceTokenizer()
        text=data[data["label"] == i[0]]
        column_text="text"
        quantity=20 
        all_words = ' '.join([text for text in text[column_text]])
        token_phrase = token_space.tokenize(all_words)
        frequency = nltk.FreqDist(token_phrase)
        
        datacount[i[0]]['frequency']=[] 
        
        df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
        df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity) 
                
        for index,d in df_frequency.iterrows():
             datacount[i[0]]['frequency'].append({'word':d["Word"],'frequency':d["Frequency"]})
             
        ax = seaborn.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
        ax.set(ylabel = "Count")
        plt.xticks(rotation='vertical')
        plt.savefig(settings.STATIC_ROOT+'\\'+i[0]+'-wordcount.png') 
        plt.close()
        
    context={'datacount':datacount.items()}
    
    return render(request, 'explorewordcount.html', context=context)      
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})  

def plot_confusion_matrix(atype,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    thresh = cm.max() / 2.
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black") 
                 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize: 
       plt.savefig(settings.STATIC_ROOT+'\\confusion_matrix_with_normalize_'+atype+'.png') 
    else:
       plt.savefig(settings.STATIC_ROOT+'\\confusion_matrix_without_normalize_'+atype+'.png') 
    plt.close()

def exploreconfusionmatrix(request,atype):
  try:
    df= pd.read_csv(filename) 
    data=cleandata(df)    
    dcount=data.groupby(['label'])['label'].agg(list)
    datacount={}
    for i in dcount:
        datacount[i[0]]=len(i)
    
    context={'datacount':datacount.items()}
    
    context['atype']=atype
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=42) 
    if atype=="lr": 
      #Logistic regression
      context['atypelabel']="Logistic regression"
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])# Fitting the model
                 
      
       
      model = pipe.fit(X_train, y_train)# Accuracy
      prediction = model.predict(X_test) 
      context['accuracy']= format(round(metrics.accuracy_score(y_test, prediction)*100,2))+"%" 
    
      cm = metrics.confusion_matrix(y_test, prediction) 
    
    elif atype=="dtc":
      context['atypelabel']="Decision Tree Classifier"
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
      # Fitting the model
      model = pipe.fit(X_train, y_train)# Accuracy
      prediction = model.predict(X_test)
      context['accuracy']=format(round(metrics.accuracy_score(y_test, prediction)*100,2))
 
      cm = metrics.confusion_matrix(y_test, prediction)    
      
    elif atype=="rfc":
      context['atypelabel']="Random Forest Classifier"    
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
      model = pipe.fit(X_train, y_train)
      prediction = model.predict(X_test)
      context['accuracy']=format(round(metrics.accuracy_score(y_test, prediction)*100,2))
      cm = metrics.confusion_matrix(y_test, prediction) 
      
    plot_confusion_matrix(atype,cm, classes=['Fake', 'Real']) 
    
    return render(request, 'exploreconfusionmatrix.html', context=context)              
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})
    
def exploreconfusionmatrixnormlize(request,atype):
  try:
    df= pd.read_csv(filename)  
    data=cleandata(df)    
    dcount=data.groupby(['label'])['label'].agg(list)
    datacount={}
    for i in dcount:
        datacount[i[0]]=len(i)
    
    context={'datacount':datacount.items()}    
    context['atype']=atype
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.label, test_size=0.2, random_state=42) 
    
    if atype=="lr": 
      #Logistic regression
      context['atypelabel']="Logistic regression"
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])# Fitting the model
                 
      
       
      model = pipe.fit(X_train, y_train)# Accuracy
      prediction = model.predict(X_test) 
      context['accuracy']= format(round(metrics.accuracy_score(y_test, prediction)*100,2))+"%" 
    
      cm = metrics.confusion_matrix(y_test, prediction) 
    
    elif atype=="dtc":
      context['atypelabel']="Decision Tree Classifier"
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
      # Fitting the model
      model = pipe.fit(X_train, y_train)# Accuracy
      prediction = model.predict(X_test)
      context['accuracy']=format(round(metrics.accuracy_score(y_test, prediction)*100,2))
 
      cm = metrics.confusion_matrix(y_test, prediction) 
      
    elif atype=="rfc":
      context['atypelabel']="Random Forest Classifier"    
      pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
      model = pipe.fit(X_train, y_train)
      prediction = model.predict(X_test)
      context['accuracy']=format(round(metrics.accuracy_score(y_test, prediction)*100,2))
      cm = metrics.confusion_matrix(y_test, prediction) 
    
    plot_confusion_matrix(atype,cm, classes=['Fake', 'Real'],normalize=True) 
    
    return render(request, 'exploreconfusionmatrixnormlize.html', context=context)     
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})    