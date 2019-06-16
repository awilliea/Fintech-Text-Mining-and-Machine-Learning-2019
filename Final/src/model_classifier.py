import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_score(df,outputfile):
    emotion=['Negative','Positive','Uncertainty','Litigious','StrongModal','WeakModal','Constraining']
    emotion=pd.DataFrame(columns=emotion)
    for i in range(df.shape[0]):
        try:
            emotion.loc[i]=df["score"].values[i][1:-1].split(",")
        except:
            emotion.loc[i]=0
    emotion=emotion.astype(float)
    emotion.index=df.index
    score_combine_emotion=pd.concat([df,emotion],axis=1)
    score_combine_emotion.to_csv(outputfile)
    
    return score_combine_emotion
def data_preprocessing(score_combine_emotion):
    X=score_combine_emotion[['Negative','Positive', 'Uncertainty', 'Litigious', 'StrongModal', 'WeakModal','Constraining']]
    y=score_combine_emotion["bad"]
    desire_index=pd.concat([X,y],axis=1)[(X.sum(axis=1)>100)].index
    X=X.loc[desire_index].values
    y=y.loc[desire_index].values
    linearly_separable = (X, y)
    multiple=int(len(y)/len(y[y==1]))
    
    # enlarge the bad companies
    X_adj=list(X)+list(X[y==1])*8
    y_adj=list(y)+list(y[y==1])*8
    
    # normalize 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    
    # normalize adj
    X_adj = StandardScaler().fit_transform(X_adj)
    X_train_adj, X_test_adj, y_train_adj, y_test_adj = train_test_split(X_adj, y_adj, test_size=.4, random_state=42)
    
    return X_train, X_test, y_train, y_test, X_train_adj, X_test_adj, y_train_adj, y_test_adj 

def train(X_train,X_test,y_train,y_test,adj=False,extra = False):
    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    log_val = pd.DataFrame(columns=log_cols)
    
    conf_dic = {}
    conf_dic_val = {}
    
    classifiers_name = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier','GradientBoostingClassifier',\
    'GaussianNB','LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']
    
    best_acc = 0
        
    for i,clf in enumerate(classifiers):
        try:
            # train
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__

            print("="*30)
            print(name)

            print('****Results****')
            train_predictions = clf.predict(X_train)
            acc = accuracy_score(y_train, train_predictions)
            conf_mat = confusion_matrix(y_train, train_predictions)
            conf_dic[name] = conf_mat
            
            print("Accuracy: {:.4%}".format(acc))
            print(conf_mat)
            train_predictions = clf.predict_proba(X_train)
            
            ll = log_loss(y_train, train_predictions)
            print("Log Loss: {}".format(ll))
            
            log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
            log = log.append(log_entry)
        
            # test
            test_predictions = clf.predict(X_test)
            acc_test = accuracy_score(y_test, test_predictions)
            conf_mat_test = confusion_matrix(y_test, test_predictions)
            conf_dic_val[name] = conf_mat_test
            
            test_predictions = clf.predict_proba(X_test)
            ll_test = log_loss(y_test, test_predictions)
            print("Log Loss: {}".format(ll_test))

            log_entry_test = pd.DataFrame([[name, acc_test*100, ll_test]], columns=log_cols)
            log_val = log_val.append(log_entry_test)
            
            if(acc_test > best_acc):
                best_acc = acc_test
                best_model = clf
                best_log = log_entry_test
                best_conf = conf_mat_test
                best_model_name = classifiers_name[i]
        except:
#             break
            continue

    print("="*30)
    
    # Save the model
    model_name = f'./Model/{best_model_name}_model'
    log_name = f'./Model/{best_model_name}_log'
    conf_name = f'./Model/{best_model_name}_conf'
        
    if (extra):
        model_name += '_extra'
        log_name += '_extra'
        conf_name += '_extra'
        if(adj):
            model_name += '_adj'
            log_name += '_adj'
            conf_name += '_adj'
    else:
        if(adj):
            model_name += '_adj'
            log_name += '_adj'
            conf_name += '_adj'
            
    model_name += '.pkl'
    log_name += '.pkl'
    conf_name += '.pkl'
    
    
    if(not os.path.isdir('./Model')):
        os.mkdir('./Model')
        
    with open(model_name,'wb') as f:
        pickle.dump(best_model,f)
    with open(log_name,'wb') as f:
        pickle.dump(best_log,f)
    with open(conf_name,'wb') as f:
        pickle.dump(best_conf,f)
    
    return log,conf_dic,log_val,conf_dic_val

if __name__ == '__main__':
    
    # Load files
    df = pd.read_csv("../score_QK.csv",index_col=0)
    score_combine_emotion = split_score(df,"score_combine_emotion_QK.csv")
    
    # Model
    classifiers_name = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier','GradientBoostingClassifier',\
    'GaussianNB','LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']
    
    classifiers = [
    KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025, probability=True),
#     NuSVC(probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
    
    X_train, X_test, y_train, y_test, X_train_adj, X_test_adj, y_train_adj, y_test_adj = data_preprocessing(score_combine_emotion)
    X_train_e, X_test_e, y_train_e, y_test_e, X_train_adj_e, X_test_adj_e, y_train_adj_e, y_test_adj_e = data_preprocessing(score_combine_emotion_extra)
    
    # Train
    log,conf_dic,log_val,conf_dic_val = train(X_train,X_test,y_train,y_test)
    log_adj,conf_dic_adj,log_val_adj,conf_dic_val_adj = train(X_train_adj, X_test_adj, y_train_adj, y_test_adj,adj=True)
    log_e,conf_dic_e,log_val_e,conf_dic_val_e = train(X_train_e, X_test_e, y_train_e, y_test_e,extra=True)
    log_adj_e,conf_dic_adj_e,log_val_adj_e,conf_dic_val_adj_e = train(X_train_adj_e, X_test_adj_e, y_train_adj_e, y_test_adj_e,extra=True,adj=True)
    
    
    