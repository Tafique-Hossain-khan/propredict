from flask import Flask, render_template, request
from src.exception import CustomException
from src.logger import logging
import sys
#from src.phone_price.pipeline.traning_pipeline import TraningPhone
from src.utils import load_object
from src.phone_price.pipeline.prediction_pipeline import CustomInput
from src.laptop_price.pipeline.prediction_pipeling import CustomInputLaptop
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-phone', methods=['GET', 'POST'])
def predict_phone():
    try:
        if request.method == 'POST':
            # Handle form submission for phone prediction
            dispaly_res_type = request.form.get('display_res_type')
            os = request.form.get("os")
            processor_brand  = request.form.get('processor_brand')
            connectivity_feature  = int(request.form.get('connectivity_feature'))
            ram  = int(request.form.get('ram'))
            internal_storgar  = request.form.get('internal_storage')
            mobile_brand  = request.form.get('mobile_brand')
            dispaly_size_in_inches  = int(request.form.get('display_size_in_inches'))
            back_camera  = request.form.get('back_camera')
            front_camera  = request.form.get('front_camera')
            Battery_Category  = request.form.get('battery_category')
        
            
            #load the trained model and then do the prediction
            preprocesoor = load_object('artifacts/phone/preprocessor.pkl')
            model = load_object('artifacts/phone/model.pkl')
            logging.info("Model loaded")
            ci = CustomInput(dispaly_res_type	,os,	processor_brand,	connectivity_feature,	ram,	internal_storgar	
                ,mobile_brand,	dispaly_size_in_inches,	back_camera,	front_camera,	Battery_Category)
            df = ci.custome_dataset()
            logging.info(df.head())
            scaled_data = preprocesoor.transform(df)
            logging.info("Data scaling done")
            #pred = PredictPipeline()
            prediction = model.predict(scaled_data)
            #prediction_result = f"Display Type: {display_res_type}"
            #print(display_res_type)
            #return "This code works"
            return render_template('phone_form.html', prediction_result=prediction)
        return render_template('phone_form.html')
    except Exception as e:
        raise CustomException (e,sys)

@app.route('/predict-laptop', methods=['GET', 'POST'])
def predict_laptop():
    try:
        if request.method == 'POST':
            # Handle form submission for laptop prediction
            company = request.form.get('company')
            TypeName = request.form.get('typeName')
            Ram = int(request.form.get('ram'))
            Weight = float(request.form.get('weight'))
            Touchscreen = int(request.form.get('touchscreen'))
            IPS = int(request.form.get('ips'))
            ppi = float(request.form.get('ppi'))
            Cpu_brand = request.form.get('cpu_brand')
            HDD = int(request.form.get('hdd'))
            SSD = int(request.form.get('ssd'))
            Gpu_brand = request.form.get('gpu_brand')
            os = request.form.get('os')

            #get the modela and the preprocessor
            preprocessor_laptop = load_object('artifacts/laptop/preprocessor_laptop.pkl')
            model_laptop = load_object('artifacts/laptop/model_laptop.pkl')
            logging.info("Model loaded")
            #ci = CustomInput('MSI','Gaming',16,2.43,0,1,'Intel Core i7',1000,256,'Nvidia'	,'Windows')
            ci1 = CustomInputLaptop(company, TypeName, Ram, Weight, Touchscreen, IPS,ppi, Cpu_brand, HDD, SSD, Gpu_brand, os)
            df1 = ci1.custom_dataset()
            logging.info(df1)
            scaled_data = preprocessor_laptop.transform(df1)
            logging.info("Data scaling done for laptop")
            #pred = PredictPipeline()
            prediction = model_laptop.predict(scaled_data)

        
            return render_template('laptop_form.html', prediction_result=prediction)
        return render_template('laptop_form.html')
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(debug=True)
