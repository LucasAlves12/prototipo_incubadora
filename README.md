# prototipo_incubadora


# Importando as bibliotecas necessárias
import numpy as np
from picamera2 import Picamera2
import pygame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import time

# Configuração de diretórios
base_dir = '//home//matheus//prototipo_incubadora'  # Substituir pelo caminho onde estão as imagens
classes = ['perfeito', 'defeituoso']

# Leitura e preprocessamento de imagens
def carregar_imagens(base_dir, classes, img_size=(128, 128)):
    imagens = []
    labels = []
    for idx, classe in enumerate(classes):
        dir_classe = os.path.join(base_dir, classe)
        for arquivo in os.listdir(dir_classe):
            caminho_arquivo = os.path.join(dir_classe, arquivo)
            imagem = cv2.imread(caminho_arquivo)
            imagem = cv2.resize(imagem, img_size)
            imagens.append(imagem)
            labels.append(idx)
    imagens = np.array(imagens) / 255.0  # Normalização
    labels = to_categorical(labels, num_classes=len(classes))  # One-hot encoding
    return imagens, labels

# Carregando os dados
imagens, labels = carregar_imagens(base_dir, classes)
X_train, X_test, y_train, y_test = train_test_split(imagens, labels, test_size=0.2, random_state=42)

# Construção do modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')  # Saída com o número de classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Augmentação de dados
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Treinamento
batch_size = 16
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
history = model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))

# Inicializando o Pygame
pygame.init()

# Configura a janela do Pygame
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Detecção de Defeitos - PiCamera2')

# Inicializa a PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration())
picam2.start()

# Função para converter a imagem capturada pela PiCamera2 para o formato Pygame
def convert_to_surface(image_array):
    image_array = np.flipud(image_array)  # Reverte a imagem verticalmente
    image = pygame.surfarray.make_surface(image_array)  # Converte o array numpy para superfície do Pygame
    return image

# Captura e classificação em tempo real
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False

    # Captura um frame da PiCamera2
    frame = picam2.capture_array()

    # Pré-processamento do frame
    frame_resized = cv2.resize(frame, (128, 128))  # Redimensiona para o tamanho adequado
    frame_normalized = frame_resized / 255.0  # Normaliza a imagem

    # Classificação da imagem
    prediction = model.predict(np.expand_dims(frame_normalized, axis=0))
    class_name = classes[np.argmax(prediction)]

    # Exibe o nome da classe na imagem
    cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Converte o frame da PiCamera2 para a superfície do Pygame e exibe
    frame_surface = convert_to_surface(frame)
    screen.blit(frame_surface, (0, 0))
    pygame.display.update()

    # Aguarda um pequeno tempo para a próxima atualização (cerca de 30fps)
    time.sleep(0.033)

# Finaliza e fecha a câmera e a janela do Pygame
picam2.stop()
pygame.quit()
