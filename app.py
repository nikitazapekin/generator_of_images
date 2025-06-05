import os
import time  # Добавлен отсутствующий импорт
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from models import Autoencoder
from utils import prepare_dataset, generate_image
import torch
import torch.nn as nn
import torch.optim as optim
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

'''           
# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
model.eval()
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

# Загрузка модели с проверкой существования файла
if os.path.exists('autoencoder.pth'):
    model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
else:
    print("Model weights not found, initializing new model...")
    model.apply(init_weights)
    torch.save(model.state_dict(), 'autoencoder.pth')

model.eval()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files')
    if len(files) == 0:
        return redirect(request.url)


    dataset_id = str(len(os.listdir(app.config['UPLOAD_FOLDER']))) + '_' + str(int(time.time()))
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
    os.makedirs(dataset_path, exist_ok=True)


    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(dataset_path, filename))


    dataset = prepare_dataset(dataset_path)
    torch.save(dataset, os.path.join(dataset_path, 'dataset.pt'))

    return redirect(url_for('index', dataset_id=dataset_id))


@app.route('/generate', methods=['POST'])
def generate():
    dataset_id = request.form.get('dataset_id')
    if not dataset_id:
        return redirect(url_for('index'))

    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
    dataset_file = os.path.join(dataset_path, 'dataset.pt')

    if not os.path.exists(dataset_file):
        return redirect(url_for('index'))

    # Generate image
    output_path = os.path.join(app.config['GENERATED_FOLDER'], f'generated_{dataset_id}.png')
    generate_image(model, dataset_file, output_path, device)

    return redirect(url_for('index', generated_id=dataset_id))


@app.route('/generated/<filename>')
def generated_image(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


# Добавьте новый маршрут для обучения
@app.route('/train', methods=['POST'])
def train_model():
    dataset_id = request.form.get('dataset_id')
    if not dataset_id:
        return redirect(url_for('index'))

    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
    dataset_file = os.path.join(dataset_path, 'dataset.pt')

    if not os.path.exists(dataset_file):
        return redirect(url_for('index'))

    # Загружаем датасет
    dataset = torch.load(dataset_file)

    # Параметры обучения
    epochs = 50
    batch_size = 8
    learning_rate = 0.001

    # Инициализируем модель и оптимизатор
    model = Autoencoder().to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Цикл обучения
    for epoch in range(epochs):
        # Перемешиваем данные
        indices = torch.randperm(len(dataset))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = dataset[batch_indices].to(device)

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'autoencoder.pth')
    model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
    model.eval()

    return redirect(url_for('index', message="Model trained successfully!"))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    app.run(debug=True)