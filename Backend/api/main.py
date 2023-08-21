from fastapi import FastAPI , File , UploadFile 
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO

import tensorflow as tf
# from tensorflow.keras.models import load_model

BetaModel = tf.keras.models.load_model('../../tf_model/1')
ProdModel = tf.keras.models.load_model('../../tf_model/2')

class_names = ['Early_blight', 'Late_blight', 'healthy']

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


def get_pred_img(img_array , version):
    
    if (version == 1):
        print('beta model')
        model = BetaModel
    else:
        print('prod model')
        model = ProdModel 
        
    img_batch = np.expand_dims(img_array, 0)
    pred_array = model.predict(img_batch)[0]
    
    
    confidence_level = np.round(100*np.max(pred_array) , 2)
    leaf_type = class_names[np.argmax(pred_array)]
    
    result = {"Leaf Type " : leaf_type , "Cconfidence Level " : confidence_level}
    # for label, confidence in zip(class_names , pred_array):
            # result[label] = np.round(100*confidence , 2)
    
    return result

@app.post("/predict/{version}")
async def predict( version ,   file : UploadFile = File(...)   ):   
    
    img  = await file.read()
    img_bytes = Image.open(BytesIO(img))
    img_array = np.array(img_bytes)

    result = get_pred_img(img_array , version)

    
    return result



# if __name__ =='__main__':
#     uvicorn.run(app , host='localhost' ,port = 8000)


