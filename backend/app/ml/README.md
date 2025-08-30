# Card Detection ML Training Guide

This guide will help you improve your card detection model by addressing the three main issues you identified:
1. **Insufficient training data**
2. **Poor data quality/variety**
3. **Inadequate model architecture**

## 🚀 Quick Start

For beginners, the easiest way to improve your model is to run the complete training pipeline:

```bash
# Navigate to the ML directory
cd backend/app/ml

# Install dependencies
pip install -r ../../requirements.txt

# Run the complete training pipeline
python training_pipeline.py
```

This will guide you through:
1. Data collection
2. Data augmentation
3. Transfer learning with YOLOv8
4. Model evaluation

## 📁 File Overview

### Core Training Files
- **`training_pipeline.py`** - Complete automated training pipeline (START HERE)
- **`transfer_learning.py`** - YOLOv8 transfer learning implementation
- **`data_collector.py`** - Interactive data collection tool
- **`data_augmentation.py`** - Dataset augmentation and preparation

### Original Files
- **`card_detector.py`** - Your existing OpenCV-based detector
- **`__init__.py`** - Module initialization

## 🎯 Step-by-Step Improvement Guide

### Step 1: Collect More Training Data

**Problem**: Your model needs more diverse training examples.

**Solution**: Use the data collection tool to gather 500-1000 card images.

```bash
# Run data collection tool
python data_collector.py
```

**What to collect**:
- **Variety**: All 52 card types (A-K in 4 suits)
- **Lighting**: Bright, dim, natural, artificial light
- **Backgrounds**: Different tables, fabrics, colors
- **Angles**: Straight, tilted, perspective views
- **Quantities**: Single cards, multiple cards, overlapping
- **Quality**: Clear and slightly blurry images

**Target**: 10-20 images per card type minimum

### Step 2: Augment Your Dataset

**Problem**: Even with manual collection, you need more training variety.

**Solution**: Use data augmentation to multiply your dataset 5-10x.

```bash
# Run data augmentation
python data_augmentation.py
```

**What it does**:
- **Geometric**: Rotation, scaling, perspective changes
- **Color**: Brightness, contrast, hue adjustments
- **Quality**: Noise, blur, compression effects
- **Environmental**: Shadows, lighting variations

**Result**: 500 original images → 2500+ training images

### Step 3: Use Transfer Learning

**Problem**: Training from scratch requires massive datasets and compute.

**Solution**: Use pre-trained YOLOv8 models and fine-tune for cards.

```bash
# Run transfer learning
python transfer_learning.py
```

**Model options**:
- **YOLOv8n** (nano): Fastest, least accurate
- **YOLOv8s** (small): Good balance
- **YOLOv8m** (medium): **Recommended for most users**
- **YOLOv8l** (large): Better accuracy, slower
- **YOLOv8x** (extra): Best accuracy, slowest

## 🛠️ Detailed Usage Instructions

### Data Collection

1. **Start collection**:
   ```bash
   python data_collector.py
   ```

2. **Follow prompts**:
   - Position cards clearly in camera view
   - Press SPACE to capture
   - Label each card (rank and suit)
   - Press 'q' to quit

3. **Collection tips**:
   - Collect in different rooms/lighting
   - Use various backgrounds
   - Include multiple cards per image sometimes
   - Vary distance and angles

### Data Augmentation

1. **Run augmentation**:
   ```bash
   python data_augmentation.py
   ```

2. **Configuration options**:
   - Number of augmentations per image (default: 5)
   - Train/validation/test split ratios
   - Output format (YOLO, JSON)

3. **Output structure**:
   ```
   data/augmented/
   ├── train/
   │   ├── images/
   │   ├── labels/
   │   └── annotations/
   ├── val/
   └── test/
   ```

### Transfer Learning Training

1. **Basic training**:
   ```bash
   python transfer_learning.py
   ```

2. **Advanced options**:
   ```python
   # In Python script
   from transfer_learning import CardDetectorYOLO
   
   detector = CardDetectorYOLO(model_size='m')
   
   # Customize training
   detector.hyperparams.update({
       'epochs': 150,
       'batch_size': 32,
       'patience': 75
   })
   
   # Train
   results = detector.train_model()
   ```

3. **Monitor training**:
   - Training plots saved automatically
   - Best model saved as `best.pt`
   - Validation metrics displayed

## 📊 Understanding Results

### Key Metrics

- **mAP50**: Mean Average Precision at 50% IoU threshold
  - 0.9+ = Excellent
  - 0.7-0.9 = Good
  - 0.5-0.7 = Fair
  - <0.5 = Needs improvement

- **Precision**: How many detections were correct
- **Recall**: How many actual cards were found
- **F1-Score**: Balance of precision and recall

### Training Plots

After training, check these plots in the results folder:
- **Loss curves**: Should decrease over time
- **mAP curves**: Should increase over time
- **Confusion matrix**: Shows which cards are confused
- **Prediction examples**: Visual results on test images

## 🔧 Troubleshooting

### Common Issues

**1. "No CUDA device found"**
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, training will use CPU (slower but works)
```

**2. "Out of memory"**
- Reduce batch size: `detector.hyperparams['batch_size'] = 8`
- Use smaller model: `model_size='n'` or `model_size='s'`

**3. "No annotations found"**
- Make sure you've labeled your collected images
- Check that annotation files exist in `data/annotations/`

**4. "Poor accuracy"**
- Collect more diverse training data
- Increase augmentation multiplier
- Train for more epochs
- Try larger model size

### Performance Tips

**For faster training**:
- Use smaller model (`yolov8n` or `yolov8s`)
- Reduce image size: `imgsz=416`
- Increase batch size if you have GPU memory

**For better accuracy**:
- Use larger model (`yolov8l` or `yolov8x`)
- Collect more training data
- Train for more epochs
- Use higher image resolution: `imgsz=832`

## 🎯 Expected Improvements

After following this guide, you should see:

### Before (Current OpenCV Method)
- Limited to template matching
- Poor performance in varied conditions
- No learning from new data
- Accuracy: ~60-70%

### After (YOLOv8 Transfer Learning)
- Deep learning-based detection
- Robust to lighting/angle changes
- Learns from your specific data
- Expected accuracy: 85-95%

### Timeline
- **Data collection**: 2-4 hours
- **Data augmentation**: 30 minutes
- **Model training**: 1-3 hours (depending on hardware)
- **Total improvement time**: 4-8 hours

## 🚀 Advanced Usage

### Custom Training Scripts

Create your own training script:

```python
from training_pipeline import CardDetectionTrainingPipeline

# Initialize pipeline
pipeline = CardDetectionTrainingPipeline("./my_data")

# Customize configuration
pipeline.config['model_training']['epochs'] = 200
pipeline.config['data_augmentation']['augmentations_per_image'] = 8

# Run specific steps
pipeline._run_data_augmentation()
pipeline._run_model_training('l')  # Use large model
```

### Integration with Existing Code

Replace your current detector:

```python
# Old way
from ml.card_detector import CardDetector
detector = CardDetector()

# New way
from ml.transfer_learning import CardDetectorYOLO
detector = CardDetectorYOLO()
detector.model = YOLO('path/to/your/trained/best.pt')

# Same interface
detections = detector.predict_image('image.jpg')
```

### Model Export

Export for deployment:

```python
detector = CardDetectorYOLO()
detector.export_model('best.pt', 'onnx')  # For faster inference
```

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 🤝 Getting Help

If you encounter issues:

1. **Check the error message** - most issues have clear solutions
2. **Review the troubleshooting section** above
3. **Start with the complete pipeline** - it handles most edge cases
4. **Use smaller datasets first** - test with 50-100 images initially

## 📈 Next Steps

After improving your model:

1. **Deploy the new model** in your application
2. **Collect more data** as you use the system
3. **Retrain periodically** with new data
4. **Monitor performance** and adjust as needed

Remember: Machine learning is iterative. Start with the basic pipeline, see the improvements, then gradually customize and optimize based on your specific needs!