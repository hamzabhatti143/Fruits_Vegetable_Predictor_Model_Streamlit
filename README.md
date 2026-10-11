# 🥗 Fruits & Vegetables Classifier

A Streamlit web app that identifies fruits and vegetables from a photo using a trained Keras CNN. Upload an image and get the predicted class, confidence score, and top-3 matches instantly.

## Features

- Instant prediction on image upload (JPG, JPEG, PNG, WEBP)
- Predicted item shown as a visual card (emoji by default, or your own reference photo)
- Confidence score with Fruit / Vegetable / Low-confidence badge
- Top-3 matches with progress bars
- Model cached in memory, so it loads once, not on every interaction
- Light and dark theme friendly

## Model

| Property | Value |
|---|---|
| Architecture | Sequential CNN (3× Conv2D + MaxPooling, Dense 128, Dense 36) |
| Input | 180 × 180 × 3 RGB |
| Preprocessing | `Rescaling(1/255)` is built into the model, so no manual normalization |
| Output | 36 logits (softmax applied in the app) |
| Format | `.keras` (saved with Keras 3.13.2) |

### Supported classes (36)

apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip, watermelon

> **Important:** labels are loaded from `class_names.json`, not hardcoded. It must be the exact `train_ds.class_names` from training. Regenerate it with:
> ```python
> import json
> json.dump(train_ds.class_names, open('class_names.json', 'w'))
> ```
> The app refuses to start if the label count does not match the model's output size.

## Project Structure

```
Fruits_Veg_dataset/
├── app.py
├── class_names.json         # class labels, in training order
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   └── Fruits_Veg_Model_Data.keras
└── reference_images/        # optional: apple.jpg, banana.png, ...
```

## Setup

Requires Python 3.10-3.12.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app opens at http://localhost:8501.

## Reference Images (optional)

By default the prediction card shows an emoji. To show a real photo instead, add images named after the class to `reference_images/`:

```
reference_images/apple.jpg
reference_images/bell pepper.png
```

Supported extensions: `jpg`, `jpeg`, `png`, `webp`. Classes without an image fall back to the emoji.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the model is ~48 MB, under GitHub's 100 MB file limit).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and select `app.py`.
3. If the install fails, match the TensorFlow/Keras versions used for training and pin them in `requirements.txt`.

## Troubleshooting

| Issue | Fix |
|---|---|
| Wrong labels, high confidence | Regenerate `class_names.json` from `train_ds.class_names` |
| `load_model` version error | Pin the same `tensorflow` / `keras` versions as training |
| Low accuracy on your photos | Use a clear, well-lit photo with a single item in frame |
| TensorFlow info/warning logs in terminal | Harmless (oneDNN, CPU instructions, no GPU on Windows) |

## Tech Stack

Python · Streamlit · TensorFlow / Keras · NumPy · Pillow
