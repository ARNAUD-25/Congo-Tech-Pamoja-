from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('get_resnext101_weighted.pth', map_location=device)
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_index = predicted.item()
    class_name = ["Healthy", "Diseased"][class_index]  
    return {"class_index": class_index, "class_name": class_name}
