from flask import Flask, render_template, request
from src.exception import CustomException
from src.logger import logging
import sys
#from src.phone_price.pipeline.traning_pipeline import TraningPhone
from src.utils import load_object
from src.phone_price.pipeline.prediction_pipeline import CustomInput
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

            logging.info(dispaly_res_type)
            logging.info(os)
            logging.info(processor_brand)
            logging.info(connectivity_feature)
            logging.info(ram)
            logging.info(internal_storgar)
            logging.info(mobile_brand)
            logging.info(f"display size:{dispaly_size_in_inches}")
            logging.info(back_camera)
            logging.info(front_camera)
            logging.info(f"battery categoty{Battery_Category}")
            
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
    if request.method == 'POST':
        # Handle form submission for laptop prediction
        return "Laptop prediction form submitted"
    return render_template('laptop_form.html')

if __name__ == '__main__':
    app.run(debug=True)
