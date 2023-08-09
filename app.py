from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.transforms import functional as T
from typing import List
import uvicorn
import cv2
import io

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import pkg_resources

for dist in pkg_resources.working_set:
    print(dist.project_name, dist.version)
app = FastAPI()
origins = ["http://localhost", "http://localhost:4200"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.mount("/public", StaticFiles(directory="./public"), name="public")




class MobileNetV2MultiClass(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2MultiClass, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Número de clases en la clasificación multi-clase (por ejemplo, 4 clases: 'bueno', 'malo', 'regular' e 'indeterminado')
num_classes = 4

# Crear una instancia del modelo MobileNetV2 para clasificación multi-clase
model = MobileNetV2MultiClass(num_classes)

# Crear una instancia del modelo MobileNetV2 para clasificación multi-label


# Cargar los pesos del modelo entrenado
# Para cargar el modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./model/best_model3.pth", map_location=device))
model.eval()
# Mover el modelo al dispositivo (GPU si está disponible)
#model = model.to(device)

# Define class labels
class_labels = ['Sana', 'Inicial', 'Intermedia', 'Final']


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse('./public/index.html')

# ...

@app.post("/prediction")
async def predict(files: List[UploadFile] = File(...)):
    result = dict()
    images = []
    
    try:
        for file in files:
            uploaded_image = await file.read()
            uploaded_image = Image.open(io.BytesIO(uploaded_image))
          # arr = to_arr(img, cv2.IMREAD_COLOR)
          # arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
          # arr = arr / 255
           # Aplicar las transformaciones necesarias para que coincidan con las que se usaron durante el entrenamiento
            # Asegurarse de que la imagen sea cuadrada (en este caso, 240x240)
            uploaded_image = T.resize(uploaded_image, 240)
            uploaded_image = T.center_crop(uploaded_image, 240)

            # Convertir la imagen en un tensor y normalizarla
            uploaded_image = T.to_tensor(uploaded_image)
            uploaded_image = T.normalize(uploaded_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Agregar una dimensión adicional para el batch (el modelo espera un batch de imágenes)
            uploaded_image = uploaded_image.unsqueeze(0)

            # Mover la imagen al dispositivo (GPU si está disponible)
            uploaded_image = uploaded_image.to(device)
            images.append(uploaded_image)
                        

            #images = np.array(images)

        # Convertir las imágenes numpy a tensores de PyTorch y moverlos al dispositivo
        # Aplicar las transformaciones necesarias para que coincidan con las que se usaron durante el entrenamiento
        # Asegurarse de que la imagen sea cuadrada (en este caso, 240x240)
        
        # Realizar la inferencia
        y = model(uploaded_image)
        print(y)
        for i in range(len(y)):
            
            predicted_class = class_labels[np.argmax(y[i].cpu().detach().numpy())]  # Convertir de tensor a numpy array
            result[files[i].filename] = predicted_class
        print(result)
        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        pass
        
    return JSONResponse(content={"message": "Error"})

# ...


def to_arr(img, cv2_img_flag=0):
    img_array = np.asarray(bytearray(img), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
