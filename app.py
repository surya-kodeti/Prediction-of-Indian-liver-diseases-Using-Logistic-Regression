import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    data_entered = request.form.to_dict()
    print(data_entered)
    
    Age = data_entered['age']
    Gender = data_entered['gender']
    if Gender == 'male' or 'Male' or 'MALE':
        Gender = int(1)
    elif Gender == 'female' or 'Female' or 'FEMALE':
        Gender = int(0)
    Total_Bilirubin = float(data_entered['total_bilirubin'])
    Direct_Bilirubin= float(data_entered['direct_bilirubin'])
    Alkaline_Phosphotase= float(data_entered['alkaline_phosphotase'])
    Alamine_Aminotransferase= float(data_entered['alamine_aminotransferase'])
    Aspartate_Aminotransferase= float(data_entered['aspartate_aminotransferase'])
    Total_Protiens= float(data_entered['total_protiens'])
    Albumin= float(data_entered['albumin'])
    Albumin_and_Globulin_Ratio= float(data_entered['albumin_and_globulin_ratio'])
    
  
    final_features = [[Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]]
    print(final_features)
    import numpy as np
    final_features = np.array(final_features,dtype=float)
    print(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    ans = " "
    if output == 2:
        ans = "not present"
    elif output == 1:
        ans = "present"
    else:
        ans = " "
        

    return render_template('index.html', prediction_text='Liver Disease Status :  {}'.format(ans))
    
   

if __name__ == "__main__":
    app.run(port = '1234',debug=True)