from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

app = Flask(__name__)

# -------------------------------
# Load models safely
# -------------------------------
def safe_load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: Model file not found: {path}")
        return None
    try:
        if path.endswith('.h5'):
            return load_model(path)
        elif path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return joblib.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# -------------------------------
# Load all models
# -------------------------------
import os
print("CURRENT WORKING DIRECTORY:", os.getcwd())

models = {}
try:
    models['feature_extractor'] = load_model('models/feature_extractor.h5')
    print("Feature extractor loaded successfully ✅")
except Exception as e:
    print("Error loading feature extractor:", e)
    models['feature_extractor'] = None
try:
    # 🩸 SUGAR MODEL
    models['diabetes'] = joblib.load(open('models/xgb_model.h5', 'rb'))
    models['scaler_diabetes'] = joblib.load('models/scalerdiab.h5')
    models['encoder_diabetes'] = joblib.load('models/label_encoderdiab.h5')
    print("Diabetes models loaded successfully ✅")
except Exception as e:
    print("Error loading diabetes models:", e)

try:
    # 🤰 PREGNANCY MODEL
    models['pregnancy'] = joblib.load(open('models/pxgboost_model.h5', 'rb'))
    models['scaler_pregnancy'] = joblib.load('models/scalerpx_model.h5')
    print("Pregnancy models loaded successfully ✅")
except Exception as e:
    print("Error loading pregnancy models:", e)


try:
    # 👁️ RETINO MODELS (same style as pregnancy)
    models['retino_2class'] = joblib.load(open('models/RF2Class.pkl', 'rb'))
    models['retino_4class'] = joblib.load(open('models/RF4Class.pkl', 'rb'))
    models['encoder_retino'] = joblib.load(open('models/label_encoder.pkl', 'rb'))

    print("Retino models loaded successfully ✅")
except Exception as e:
    print("Error loading Retiono models:", e)
        

    


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            data = [[float(request.form.get(col)) for col in [
    'cholesterol', 'glucose', 'hdl_chol', 'chol_hdl_ratio', 'age', 'gender',
    'weight', 'height', 'bmi', 'systolic_bp', 'diastolic_bp', 'waist', 'hip'
]]]
            data = np.array(data).reshape(1, -1)
            scaler = models['scaler_diabetes']
            model = models['diabetes']
            if scaler:
                data = scaler.transform(data)
            pred = model.predict(data)
            encoder = models['encoder_diabetes']
            result = encoder.inverse_transform(pred)[0] if encoder is not None else str(pred[0])
            return render_template('result_diabetes.html', prediction=result, title='Diabetes Diagnosis')
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('form_diabetes.html')

@app.route('/pregnancy', methods=['GET', 'POST'])
def pregnancy():
    if request.method == 'POST':
        try:
            data = [float(request.form.get(col)) for col in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
            data = np.array(data).reshape(1, -1)
            scaler = models['scaler_pregnancy']
            model = models['pregnancy']
            if scaler:
                data = scaler.transform(data)
            pred = model.predict(data)
            result = 'Diabetic' if pred[0] > 0.5 else 'Not Diabetic'
            return render_template('result_pregnancy.html', prediction=pred[0])

        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('form_pregnancy.html')

@app.route('/retinopathy', methods=['GET', 'POST'])
def retinopathy():
    if request.method == 'POST':
        # التحقق من رفع الصورة
        if 'image' not in request.files:
            return render_template('error.html', error='No image uploaded')
        
        img_file = request.files['image']
        if img_file.filename == '':
            return render_template('error.html', error='No image selected')
        
        # حفظ الصورة في مجلد
        img_path = os.path.join('static', 'uploaded_images', img_file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img_file.save(img_path)
        
        try:
            # تحميل ومعالجة الصورة
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # التنبؤ باستخدام نموذج 2 فئات
            model_2class = models['retino_2class']
            model_4class = models['retino_4class']
            encoder = models['encoder_retino']
            
            pred_2 = model_2class.predict(img_array)
            result_2 = np.argmax(pred_2, axis=1)[0]
            
            if result_2 == 0:
                result = 'No Diabetic Retinopathy Detected'
            else:
                # إذا تم الكشف عن DR، استخدام نموذج 4 فئات لتحديد الشدة
                pred_4 = model_4class.predict(img_array)
                result_4 = np.argmax(pred_4, axis=1)[0]
                if encoder is not None:
                    result = encoder.inverse_transform([result_4])[0]
                else:
                    result = f'Severity Level: {result_4}'
            
            return render_template('result.html', prediction=result, title='Diabetic Retinopathy Result')
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    # GET request
    return render_template('form_retino.html')


# طباعة حالة تحميل النماذج
print("\n=== MODEL LOAD STATUS ===")
for name, model in models.items():
    print(f"{name}: {'Loaded ✅' if model is not None else 'Not Loaded ❌'}")
print("==========================\n")

if __name__ == '__main__':
    app.run(debug=False)