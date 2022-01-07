import numpy as np
import pandas as pd
#import preprocess
from flask import Flask, request, render_template
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict(): 
    
    global final_string
    final_string = []

    
    def clean_punc(word): 
        cleaned = re.sub(r'[?|!|\'|#]', r'', word)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r'', cleaned)
        return cleaned

    s = ''

    def text_preprocess(text_data): 
        vect = CountVectorizer(stop_words='english')
        stop = list(vect.get_stop_words())
    
        global final_string
        global s
    
        for sentence in text_data:
            filtered_sentence = []
            for word in sentence.split():
                for cleaned_word in clean_punc(word).split():
                    if (cleaned_word.isalpha() and (len(cleaned_word) > 1) and cleaned_word not in stop):
                        s = cleaned_word.lower()
                        filtered_sentence.append(s)
                    else:
                        continue
                    
                
        strl = ""' '.join(filtered_sentence)
        wordnet = WordNetLemmatizer()
        final_string.append("".join([wordnet.lemmatize(word) for word in strl]))
        
    
        return final_string
    

    symp = request.form["SYMPTOMS POST VACCINATION"]
    text_preprocess(list(symp.split(',')))
    

    x_s = final_string

    cvect = CountVectorizer(max_features= 1000)
    cvect.fit(pd.Series(x_s))
    X_ct = cvect.transform(x_s)
    X_ct_arr = X_ct.toarray().flatten()
    X_ct_arr
    
    
    vax_typ = request.form["VAX_TYPE"]
    vax_manu = request.form["VAX_MANU"]
    vax_dose = request.form["VAX_DOSE_SERIES"]
    state = request.form["STATE"]
    gend = request.form["SEX"]
    age = request.form["AGE_GROUP"]
    vax_ad = request.form["V_ADMINBY"]
    n_days = request.form["NUMDAYS BETWEEN VAX_DATE & ONSET_DATE"]


    ls = [] 
    ls.append(float(vax_typ)) 
    ls.append(float(vax_manu)) 
    ls.append(float(vax_dose)) 
    ls.append(float(state)) 
    ls.append(float(gend))
    ls.append(float(age))
    ls.append(float(vax_ad)) 
    ls.append(float(n_days)) 
        
    ls_arr = np.array(ls)
    ls_arr
    
    final_features = np.concatenate([ls_arr, X_ct_arr])

    
    while len(final_features) != 1008: 
        final_features = np.append(final_features, 0)

    '''
    For rendering results on HTML GUI
    '''
    
    prediction = model.predict(final_features.reshape(1,-1))

    if int(prediction) == 0: 
        output = 'No_Adverse_Effect' 
    elif int(prediction) == 1: 
        output = 'Birth_Defect' 
    elif int(prediction) == 2: 
        output = 'Disability' 
    elif int(prediction) == 3: 
        output = 'Clinic_Visit' 
    elif int(prediction) == 4: 
        output = 'Hospitalization' 
    elif int(prediction) == 5: 
        output = 'Prolonged_Hospitalization' 
    elif int(prediction) == 6: 
        output = 'ER_Visit' 
    elif int(prediction) == 7: 
        output = 'Life_Threat' 
    else: 
        output= 'Death' 

    return render_template('index.html', prediction_text='Adverse Effect could be: {}'.format(output))





if __name__ == "__main__":
    app.run(debug=True)