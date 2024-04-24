import io
import json

from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
import os
from joblib import dump, load
import ssl
import coremltools as ct
# import onnx
# from onnx_coreml import convert
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

with open('plantnet300K_species_names.json', 'r') as f:
    species_names = json.load(f)


true_names = load('true_names.joblib')

model_plant = models.resnet50(weights='DEFAULT')
num_ftrs = model_plant.fc.in_features
model_plant.fc = nn.Linear(num_ftrs, 540)
model_plant.load_state_dict(torch.load('model_plant_best_accuracy.pth', map_location=torch.device('mps')))
model_plant.eval()
# example_input = torch.rand(1, 3, 256, 256)
# traced_model = torch.jit.trace(model, example_input)
# coreml_model = ct.convert(
#     traced_model,
#     convert_to="mlprogram",
#     inputs=[ct.ImageType(shape=example_input.shape)]
#  )
# coreml_model = ct.convert(
#         model,
#         inputs=[ct.ImageType(name='input', shape=(3, 256, 256))]
# )

# coreml_model.save('newmodel3.mlpackage')

# Загрузка и экспорт модели из PyTorch в ONNX
# model = models.resnet50(pretrained=True)
# dummy_input = torch.randn(1, 3, 256, 256)
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
#
# # Конвертация ONNX в Core ML
# onnx_model = onnx.load("model.onnx")
# mlmodel = convert(onnx_model)
# mlmodel.save("model.mlmodel")

model_disease = models.resnet50(weights='DEFAULT')
num_disease_ftrs = model_disease.fc.in_features
model_disease.fc = nn.Linear(num_disease_ftrs, 9)
model_disease.load_state_dict(torch.load('model_disease_best_accuracy.pth', map_location=torch.device('mps')))
model_disease.eval()

def preprocess_disease_image(image):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    # print(len(image[3]))
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    # print(len(image[3]))
    return image
# print(image)
#     print(len(image))
#     print(len(image[0]))
#     print(len(image[0][0]))
#     print(len(image[0][0][0]))
#     print(image[0][0][0][0])

disease_names = ["bacterial_blight",
                 "black_rot",
                 "healthy",
                 "late_blight",
                 "mosaic",
                 "powdery_mildev",
                 "rust",
                 "spot",
                 "yellowish"]

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        print("Изображение не найдено в запросе")
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    print("Изображение получено")

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except IOError:
        return jsonify({'error': 'Invalid image format'}), 400

    image = preprocess_disease_image(image)

    with torch.no_grad():
        outputs = model_disease(image)
        a, preds = torch.max(outputs, 1)
        predicted_disease_id = preds.item()
        predicted_disease = disease_names[predicted_disease_id]

    return jsonify({'disease': predicted_disease})


@app.route('/predict/plant', methods=['POST'])
def predict():
    print("Получен запрос")

    print(request.files)
    if 'image' not in request.files:
        print("Изображение не найдено в запросе")
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    print("Изображение получено")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        print("Изображение открыто")
    except IOError:
        print("Неверный формат изображения")
        return jsonify({'error': 'Invalid image format'}), 400

    image = preprocess_image(image)
    print("Изображение предобработано")

    with torch.no_grad():
        outputs = model_plant(image)
        print(outputs[0][702])
        print(len(outputs[0]))
        a, preds = torch.max(outputs, 1)
        print(a)
        print(preds)
        predicted_class = preds.item()
        print('Predicted class:', predicted_class)
        class_id_str = true_names[predicted_class]
        print(class_id_str)
        real_name = species_names[class_id_str]
        print("Предсказание выполнено")

    print("Отправка ответа")
    return jsonify({'class': predicted_class, 'real_name': real_name})

if __name__ == '__main__':
    app.run(host ="0.0.0.0", debug=True, port=8000)
#%%