from django.shortcuts import render
import numpy as np
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Create your views here.

model =  joblib.load(r'D:\DPS Lab\project\Django-ml-Healthcare-Prediction-Website-main\saved models\parkinsons_model.pkl')
scaler = joblib.load(r'D:\DPS Lab\project\Django-ml-Healthcare-Prediction-Website-main\saved models\parkinsons_scalar.pkl')

def index(request):
    return render(request,'index.html')

def diabetes(request):
    """ 
    Reading the training data set. 
    """
    dfx = pd.read_csv('data/Diabetes_XTrain.csv')
    dfy = pd.read_csv('data/Diabetes_YTrain.csv')
    X = dfx.values
    Y = dfy.values
    Y = Y.reshape((-1,))

    """ 
    Reading data from user. 
    """
    value = ''
    if request.method == 'POST':

        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        bloodpressure = float(request.POST['bloodpressure'])
        skinthickness = float(request.POST['skinthickness'])
        bmi = float(request.POST['bmi'])
        insulin = float(request.POST['insulin'])
        pedigree = float(request.POST['pedigree'])
        age = float(request.POST['age'])

        user_data = np.array(
            (pregnancies,
             glucose,
             bloodpressure,
             skinthickness,
             bmi,
             insulin,
             pedigree,
             age)
        ).reshape(1, 8)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, Y)

        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'Positive'
        elif int(predictions[0]) == 0:
            value = "Negative"

    return render(request,
                  'diabetes.html',
                  {
                      'context': value,
                      
                  }
                  )

def breast(request):
  
    # Reading training data set. 

    df = pd.read_csv('data/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

  
    # Reading data from user. 
    
    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

  
        # Creating our training model. 
        
        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'breast.html',
                  {
                      'context': value,
                      
                  })
# views.py

from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib

def parkinsons(request):
    result =''
    if request.method == 'POST':
        # Parse input data from the form
        fo = float(request.POST['MDVP:Fo(Hz)'])
        fhi = float(request.POST['MDVP:Fhi(Hz)'])
        flo = float(request.POST['MDVP:Flo(Hz)'])
        jitter_percent = float(request.POST['MDVP:Jitter(%)'])
        jitter_abs = float(request.POST['MDVP:Jitter(Abs)'])
        rap = float(request.POST['MDVP:RAP'])
        ppq = float(request.POST['MDVP:PPQ'])
        ddp = float(request.POST['Jitter:DDP'])
        shimmer = float(request.POST['MDVP:Shimmer'])
        shimmer_db = float(request.POST['MDVP:Shimmer(dB)'])
        apq3 = float(request.POST['Shimmer:APQ3'])
        apq5 = float(request.POST['Shimmer:APQ5'])
        apq = float(request.POST['MDVP:APQ'])
        dda = float(request.POST['Shimmer:DDA'])
        nhr = float(request.POST['NHR'])
        hnr = float(request.POST['HNR'])
        rpde = float(request.POST['RPDE'])
        dfa = float(request.POST['DFA'])
        spread1 = float(request.POST['spread1'])
        spread2 = float(request.POST['spread2'])
        d2 = float(request.POST['D2'])
        ppe = float(request.POST['PPE'])

        # Create the user data array
        user_data = np.array([
            [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
             apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        ])

        # Perform preprocessing with StandardScaler
        scaled_user_data = scaler.transform(user_data)

        # Perform prediction
        predictions = model.predict(scaled_user_data)

        if int(predictions[0]) == 1:
            result = "likely to have Parkinson's disease."
        else:
            result = "unlikely to have Parkinson's disease."

    return render(request, 'parkinsons.html', {'result': result})

