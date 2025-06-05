import os
import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
from werkzeug.utils import secure_filename
from models import Autoencoder
from utils import prepare_dataset, generate_image
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Необходимо для работы сессий
app.config['UPLOAD_FOLDER'] = 'datasets'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Инициализация модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

# Загрузка или инициализация весов модели
model_path = 'autoencoder.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("Model weights loaded successfully")
else:
    logger.info("Initializing new model weights")


    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    model.apply(init_weights)
    torch.save(model.state_dict(), model_path)

model.eval()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html',
                           dataset_id=session.get('dataset_id'),
                           generated_id=request.args.get('generated_id'),
                           message=request.args.get('message'))


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return redirect(url_for('index', error="No files selected"))

    files = request.files.getlist('files')
    if len(files) == 0 or all(file.filename == '' for file in files):
        return redirect(url_for('index', error="No files selected"))

    try:
        # Создаем уникальный ID для датасета
        dataset_id = f"{int(time.time())}_{len(files)}"
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
        os.makedirs(dataset_path, exist_ok=True)

        # Сохраняем файлы
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(dataset_path, filename))

        # Подготавливаем датасет
        dataset = prepare_dataset(dataset_path)
        if dataset is None:
            return redirect(url_for('index', error="No valid images found"))

        torch.save(dataset, os.path.join(dataset_path, 'dataset.pt'))
        session['dataset_id'] = dataset_id  # Сохраняем ID в сессии
        logger.info(f"Dataset {dataset_id} created with {len(dataset)} images")

        return redirect(url_for('index', success="Dataset created successfully"))
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return redirect(url_for('index', error=str(e)))


@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Получаем dataset_id из формы или сессии
        dataset_id = request.form.get('dataset_id', session.get('dataset_id'))
        if not dataset_id:
            return jsonify({"error": "No dataset selected"}), 400

        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
        dataset_file = os.path.join(dataset_path, 'dataset.pt')

        if not os.path.exists(dataset_file):
            return jsonify({"error": "Dataset not found"}), 404

        # Генерируем изображение
        output_filename = f'generated_{dataset_id}.png'
        output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

        if not generate_image(model, dataset_file, output_path, device):
            return jsonify({"error": "Image generation failed"}), 500

        logger.info(f"Image generated: {output_filename}")
        return jsonify({
            "success": True,
            "image_url": url_for('generated_image', filename=output_filename)
        })
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/generated/<filename>')
def generated_image(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


@app.route('/train', methods=['POST'])
def train_model():
    try:
        dataset_id = request.form.get('dataset_id', session.get('dataset_id'))
        if not dataset_id:
            return jsonify({"error": "No dataset selected"}), 400

        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_id)
        dataset_file = os.path.join(dataset_path, 'dataset.pt')

        if not os.path.exists(dataset_file):
            return jsonify({"error": "Dataset not found"}), 404

        # Загружаем датасет
        dataset = torch.load(dataset_file)
        if len(dataset) == 0:
            return jsonify({"error": "Dataset is empty"}), 400

        # Параметры обучения
        epochs = 50
        batch_size = 8
        learning_rate = 0.001

        # Инициализация модели и оптимизатора
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Обучение
        for epoch in range(epochs):
            indices = torch.randperm(len(dataset))
            for i in range(0, len(dataset), batch_size):
                batch = dataset[indices[i:i + batch_size]].to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


        torch.save(model.state_dict(), 'autoencoder.pth')
        model.eval()

        return jsonify({
            "success": True,
            "message": f"Model trained successfully with final loss {loss.item():.4f}"
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    app.run(debug=True)