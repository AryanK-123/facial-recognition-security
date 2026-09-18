from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import torch.nn.functional as f

model = InceptionResnetV1(pretrained='vggface2').eval()

def embedding(face):
    faceresized = cv2.resize(face,(160,160))
    facergb = cv2.cvtColor(faceresized, cv2.COLOR_BGR2RGB)
    facet = torch.tensor(facergb, dtype=torch.float32).permute(2,0,1) / 255.0
    facet = (facet - .5) / .5
    facet = facet.unsqueeze(0)
    with torch.no_grad():
        emb = model(facet)
    return emb

def same(e1, e2):
    dist = f.pairwise_distance(e1, e2).item()
    
    return dist < 1.15