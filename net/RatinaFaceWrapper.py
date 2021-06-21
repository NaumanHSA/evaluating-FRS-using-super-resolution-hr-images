from retinaface import RetinaFace
import cv2

def build_model():
    face_detector = RetinaFace.build_model()
    return face_detector

def detect_face(face_detector, img, align=True):

    face = None
    box = None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR
    results = RetinaFace.extract_faces(img_rgb, model=face_detector, align=align)

    if len(results) > 0:
        box = results[0]["box"]
        face = results[0]["face"][:, :, ::-1]

    return face, box
