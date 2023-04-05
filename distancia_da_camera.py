# install opencv "pip install opencv-python"
import cv2

# distance from camera to object(face) measured
# centimeter
Known_distance = 76.2

# width of face in the real world or Object Plane
# centimeter
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame

    # return the distance
    return distance

def microExpressoesFaciais(facesDetectadas, largura, altura):
    global id, confianca
    global aviso
    global reconhecedor
    global conectado, imagem
    global font
    global x
    global y
    global camera
    global imagemCinza
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classificadorEigen.yml")
    largura, altura = 220, 220
    while (True):
        conectado, imagem = camera.read()
        # Conversão em escala de cinza
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = detectorFace.detectMultiScale(imagemCinza,scaleFactor=1.5, minSize=(100, 100))
        for (x, y, l, a) in facesDetectadas:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)
            confianca = reconhecedor.predict(imagemFace)
            #Se ID é igual a 1, emite a mensagem "Seguro sair" se não, emite a mensagem "Área hostil"
            if id == 1:
                aviso = "Seguro sair"
            else:
                aviso = "Area hostil"
            aviso = cv2.putText(imagem, aviso, (x, y + (a + 30)), font, 2, (0, 0, 255))

        return aviso


def face_data(image):
    face_width = 0  # making face width to zero

    # converting color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

        # getting face width in the pixels
        face_width = w

    # return the face width in pixel
    return face_width


# reading reference_image from directory
ref_image = cv2.imread(r"C:\Users\rafae\Desktop\teste2\refimage\Ref_image.jpg")

# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image)

# get the focal by calling "Focal_Length_Finder"
# face width in reference(pixels),
# Known_distance(centimeters),
# known_width(centimeters)
Focal_length_found = Focal_Length_Finder(
    Known_distance, Known_width, ref_image_face_width)

print(Focal_length_found)

# show the reference image
cv2.imshow("ref_image", ref_image)

# initialize the camera object so that we
# can get frame from it
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# looping through frame, incoming from
# camera/video
while True:
    # reading the frame from camera
    _, frame = cap.read()

    # calling face_data function to find
    # the width of face(pixels) in the frame
    face_width_in_frame = face_data(frame)

    # check if the face is zero then not
    # find the distance
    if face_width_in_frame != 0:
        # finding the distance by calling function
        # Distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)
        if Distance <= 50:
            print("Alerta nível S!")
        elif Distance <= 100:
            print("Alerta nível A!")
        elif Distance <= 150:
            print("Alerta nível B!")
        elif Distance >= 200:
            print("Seguro sair!")

        # draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        # Drawing Text on the screen
        cv2.putText(
            frame, f"Distance: {round(Distance, 2)} CM", (30, 35),
            fonts, 0.6, GREEN, 2)

    # show the frame on the screen
    cv2.imshow("frame", frame)

    # quit the program if you press 'q' on keyboard
    if cv2.waitKey(1) == ord("q"):
        break

# closing the camera
cap.release()

# closing the windows that are opened
cv2.destroyAllWindows()
