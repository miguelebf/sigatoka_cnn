from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import sys, os
from typing import List
import uvicorn
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import pkg_resources

for dist in pkg_resources.working_set:
    print(dist.project_name, dist.version)
app = FastAPI()
origins = ["http://localhost", "http://localhost:4200", "https://clasificacion-sigatoka-cnn.web.app"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.mount("/public", StaticFiles(directory="./public"), name="public")

class MobileNetV2MultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2MultiLabel, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Número de clases en la clasificación multi-label (por ejemplo, 4 etapas: INITIAL, HEALTHY, LAST, INTERMEDIATE)
num_classes = 4

# Crear una instancia del modelo MobileNetV2 para clasificación multi-label


# Cargar los pesos del modelo entrenado
# Para cargar el modelo
model = MobileNetV2MultiLabel(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./model/best_model.pth", map_location=device))
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
            img = await file.read()
            arr = to_arr(img, cv2.IMREAD_COLOR)
            arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
            arr = arr / 255
            images.append(arr)

        images = np.array(images)

        # Convertir las imágenes numpy a tensores de PyTorch y moverlos al dispositivo
        images = torch.from_numpy(images).float()  # Convertir a tensor float32
        images = images.permute(0, 3, 1, 2)  # Cambiar el orden de las dimensiones para que sea (N, C, H, W)

        # Mover los tensores al dispositivo (GPU si está disponible)
        images = images.to(device)

        # Realizar la inferencia
        y = model(images)
        print(y)
        for i in range(len(y)):
            
            predicted_class = class_labels[np.argmax(y[i].cpu().detach().numpy())]  # Convertir de tensor a numpy array
            result[files[i].filename] = predicted_class
        print(result)
        return JSONResponse(content=result)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        pass
        
    return JSONResponse(content={"message": "Error"})

# ...


def to_arr(img, cv2_img_flag=0):
    img_array = np.asarray(bytearray(img), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
