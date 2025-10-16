# Deepfake Image Detection System

A complete AI-powered deepfake detection system using Convolutional Neural Networks (CNN) with Xception architecture.

## Features

- **CNN-Based Detection**: Uses Xception architecture with transfer learning
- **Real-time Prediction**: Upload images and get instant deepfake analysis
- **Confidence Scores**: View prediction confidence percentages
- **Prediction History**: Track all previous detections
- **Statistics Dashboard**: Monitor detection statistics
- **Modern UI**: Clean, responsive React interface
- **REST API**: FastAPI backend with MongoDB storage

## Tech Stack

### Backend
- Python 3.x
- FastAPI
- TensorFlow/Keras
- OpenCV
- MongoDB

### Frontend
- React 19
- Tailwind CSS
- Shadcn UI Components
- Axios

## Project Structure

```
/app/
├── backend/
│   ├── server.py          # FastAPI application
│   ├── train.py           # Model training script
│   ├── utils.py           # Preprocessing & Grad-CAM
│   ├── requirements.txt   # Python dependencies
│   ├── model/             # Trained model storage
│   └── dataset/           # Training data (real/fake folders)
├── frontend/
│   └── src/
│       ├── App.js         # Main React component
│       └── App.css        # Styles
└── README.md
```

## Installation

### 1. Install Backend Dependencies

```bash
cd /app/backend
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow
pip freeze > requirements.txt
```

### 2. Train the Model

```bash
cd /app/backend
python train.py
```

**Note**: The training script creates a sample dataset for demonstration. For production:
1. Download a real deepfake dataset from Kaggle (e.g., "Deepfake Detection Challenge")
2. Extract to `/app/backend/dataset/` with `real/` and `fake/` subdirectories
3. Run training again

### 3. Start Backend Server

The backend is managed by supervisor and should start automatically:

```bash
sudo supervisorctl restart backend
```

### 4. Start Frontend

```bash
cd /app/frontend
yarn start
```

## Usage

1. **Access the Web Interface**: Open your browser to the frontend URL
2. **Upload an Image**: Click the upload area or drag and drop an image
3. **Analyze**: Click "Analyze Image" to run detection
4. **View Results**: See the prediction label (REAL/FAKE) and confidence score
5. **Track History**: Check the sidebar for recent scans and statistics

## API Endpoints

### `GET /api/`
Health check endpoint

### `GET /api/model-status`
Check if model is loaded

### `POST /api/predict`
Predict if image is real or fake
- **Body**: `multipart/form-data` with `file` field
- **Returns**: Prediction result with label and confidence

### `POST /api/predict-with-gradcam`
Generate Grad-CAM visualization
- **Body**: `multipart/form-data` with `file` field
- **Returns**: Heatmap image showing model attention

### `GET /api/predictions`
Get prediction history
- **Query**: `limit` (default: 50)
- **Returns**: List of past predictions

### `GET /api/stats`
Get detection statistics
- **Returns**: Total predictions, fake/real counts, percentages

## Model Details

### Architecture
- **Base Model**: Xception (pretrained on ImageNet)
- **Custom Layers**: 
  - GlobalAveragePooling2D
  - Dense (512, 256) with ReLU activation
  - BatchNormalization
  - Dropout (0.5, 0.4)
  - Output: Dense (1) with Sigmoid

### Training Configuration
- **Input Size**: 224×224×3
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall

### Data Augmentation
- Rotation (±20°)
- Width/Height shift (±20%)
- Horizontal flip
- Brightness adjustment (0.8-1.2)

## Evaluation Metrics

The model generates:
- Training/Validation accuracy and loss curves
- Precision and Recall plots
- Confusion matrix
- Classification report (F1-score, support)

## Future Enhancements

- [ ] Real deepfake dataset integration
- [ ] Video frame-by-frame detection
- [ ] Advanced face detection (MTCNN)
- [ ] Grad-CAM visualization in UI
- [ ] Batch processing
- [ ] Model fine-tuning interface
- [ ] Export reports (PDF)

## Troubleshooting

### Model Not Loaded
- Run `python train.py` to create and train the model
- Check `/app/backend/model/deepfake_detector.h5` exists

### Prediction Errors
- Ensure image is valid (PNG, JPG, JPEG)
- Check backend logs: `tail -f /var/log/supervisor/backend.*.log`

### Training Issues
- Verify TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`
- Ensure sufficient disk space for model and dataset

## License

MIT License

## Credits

Built with FastAPI, React, TensorFlow, and MongoDB.
