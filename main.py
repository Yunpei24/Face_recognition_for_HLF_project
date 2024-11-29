from fastapi import FastAPI, File, UploadFile, HTTPException
import asyncio
from functools import wraps
import numpy as np
from PIL import Image
import io
import torch
from typing import Tuple, Optional
from facenet_pytorch import MTCNN, InceptionResnetV1
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration globale
SIMILARITY_THRESHOLD = 0.8
TIMEOUT_SECONDS = 30
MAX_WORKERS = 4

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

class FaceRecognitionService:
    def __init__(self):
        logger.info("Initialisation du service de reconnaissance faciale")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Utilisation du device: {self.device}")
        
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        try:
            logger.info("Chargement du modèle MTCNN...")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            
            logger.info("Chargement du modèle ResNet...")
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info("Modèles chargés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
            raise

    def process_image(self, img: Image.Image) -> torch.Tensor:
        """Prétraitement de l'image avec gestion des erreurs"""
        try:
            logger.debug("Début du prétraitement de l'image")
            img = img.convert('RGB')
            face = self.mtcnn(img)
            
            if face is None:
                logger.warning("Aucun visage détecté dans l'image")
                raise ValueError("Aucun visage détecté")
                
            logger.debug("Prétraitement de l'image réussi")
            return face
            
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement: {str(e)}")
            raise

    def get_face_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extraction de l'embedding avec logging détaillé"""
        try:
            logger.debug("Début de l'extraction de l'embedding")
            image = Image.open(io.BytesIO(image_bytes))
            
            # Prétraitement
            face = self.process_image(image)
            
            # Extraction de l'embedding
            with torch.no_grad():
                embedding = self.resnet(face.unsqueeze(0))
                
            logger.debug("Extraction de l'embedding réussie")
            return embedding.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de l'embedding: {str(e)}")
            return None

    async def compare_faces(self, image1_bytes: bytes, image2_bytes: bytes) -> Tuple[float, bool]:
        """Comparaison des visages avec logging"""
        logger.info("Début de la comparaison des visages")
        loop = asyncio.get_event_loop()
        
        try:
            # Traitement parallèle
            embedding1 = await loop.run_in_executor(self.thread_pool, self.get_face_embedding, image1_bytes)
            embedding2 = await loop.run_in_executor(self.thread_pool, self.get_face_embedding, image2_bytes)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Impossible de détecter les visages dans les images")
                raise HTTPException(
                    status_code=400,
                    detail="Impossible de détecter un visage dans une ou les deux images"
                )
            
            # Calcul de similarité
            similarity = float(np.dot(embedding1, embedding2) /
                            (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
            
            logger.info(f"Comparaison terminée. Similarité: {similarity}")
            return similarity, similarity > SIMILARITY_THRESHOLD
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison: {str(e)}")
            raise

app = FastAPI()
service = FaceRecognitionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/face-recognition")
@async_timeout(TIMEOUT_SECONDS)
async def face_recognition(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)) -> dict:
    """Endpoint de reconnaissance faciale avec logging"""
    logger.info("Nouvelle requête de reconnaissance faciale reçue")
    
    try:
        # Lecture des fichiers
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()
        
        logger.info("Images lues avec succès")
        
        similarity, is_similar = await service.compare_faces(image1_bytes, image2_bytes)
        
        logger.info("Comparaison terminée avec succès")
        return {
            "similarity": similarity,
            "is_similar": is_similar
        }
        
    except HTTPException as he:
        logger.error(f"HTTPException: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement des images: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "API de reconnaissance faciale"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5021, workers=4)