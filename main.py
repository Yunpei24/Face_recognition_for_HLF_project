from fastapi import FastAPI, File, UploadFile, HTTPException
import asyncio
from functools import wraps
import numpy as np
from PIL import Image
import io
import torch
from typing import Tuple
from facenet_pytorch import MTCNN, InceptionResnetV1
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Fonction decorator pour le timeout
def async_timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail="Le traitement a pris trop de temps"
                )
        return wrapper
    return decorator

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle de reconnaissance faciale
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(image: Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extrait l'embedding d'un visage à partir d'une image.
    Args:
        image (Image): L'image contenant le visage.
    Returns:
        Tuple[torch.Tensor, np.ndarray]: Le bounding box du visage et son embedding.
    """
    # Détecter et aligner le visage
    face = mtcnn(image)
    if face is None:
        raise HTTPException(status_code=400, detail="Aucun visage détecté dans l'image")
    
    # Obtenir l'embedding
    face_embedding = resnet(face.unsqueeze(0).to(device))
    return face, face_embedding.detach().cpu().numpy()[0]

@app.post("/api/face-recognition")
@async_timeout(180) 
async def face_recognition(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)) -> dict:
    """
    Endpoint pour comparer deux images de visage.
    Args:
        image1 (UploadFile): La première image de visage.
        image2 (UploadFile): La deuxième image de visage.
    Returns:
        dict: Le score de similarité et un booléen indiquant si les visages sont similaires.
    """
    try:
        # Lire les images à partir des fichiers uploadés
        img1 = Image.open(io.BytesIO(await image1.read())).convert("RGB")
        img2 = Image.open(io.BytesIO(await image2.read())).convert("RGB")

        # Extraire les embeddings des visages
        _, img1_embedding = extract_face_embedding(img1)
        _, img2_embedding = extract_face_embedding(img2)

        # Calculer la similarité cosinus entre les embeddings
        similarity = float(np.dot(img1_embedding, img2_embedding) / 
                         (np.linalg.norm(img1_embedding) * np.linalg.norm(img2_embedding)))
        
        # Définir un seuil de similarité
        is_similar = similarity > 0.8

        return {
            "similarity": similarity,
            "is_similar": is_similar
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur dans le traitement des images: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API de reconnaissance faciale"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5021, reload=True)