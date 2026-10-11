from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

MODEL_PATH = "model/Fruits_Veg_Model_Data.keras"
REF_DIR = Path("reference_images")  # optional: reference_images/apple.jpg, banana.png, ...
IMG_SIZE = (180, 180)  # from model config: input (None, 180, 180, 3)

# 36 classes, alphabetical = default order of image_dataset_from_directory.
# [unverified] Confirm against `train_ds.class_names` in your training notebook.
CLASSES = {
    "apple": "🍎", "banana": "🍌", "beetroot": "🟣", "bell pepper": "🫑",
    "cabbage": "🥬", "capsicum": "🫑", "carrot": "🥕", "cauliflower": "🥦",
    "chilli pepper": "🌶️", "corn": "🌽", "cucumber": "🥒", "eggplant": "🍆",
    "garlic": "🧄", "ginger": "🟤", "grapes": "🍇", "jalepeno": "🌶️",
    "kiwi": "🥝", "lemon": "🍋", "lettuce": "🥬", "mango": "🥭",
    "onion": "🧅", "orange": "🍊", "paprika": "🌶️", "pear": "🍐",
    "peas": "🟢", "pineapple": "🍍", "pomegranate": "🔴", "potato": "🥔",
    "raddish": "🔴", "soy beans": "🌱", "spinach": "🥬", "sweetcorn": "🌽",
    "sweetpotato": "🍠", "tomato": "🍅", "turnip": "⚪", "watermelon": "🍉",
}
CLASS_NAMES = list(CLASSES)
FRUITS = {"apple", "banana", "grapes", "kiwi", "lemon", "mango", "orange",
          "pear", "pineapple", "pomegranate", "watermelon"}

st.set_page_config(page_title="Fruits & Veg Classifier", page_icon="🥗", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {text-align:center; padding: 1.6rem 1rem; border-radius: 20px; margin-bottom: 1.5rem;
       background: linear-gradient(135deg, #16a34a 0%, #65a30d 50%, #f59e0b 100%); color: #fff;}
.hero h1 {margin: 0; font-size: 2.3rem; color: #fff;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.05rem;}
.card {border-radius: 20px; padding: 1.6rem 1rem; text-align: center;
       border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);}
.emoji {font-size: 7rem; line-height: 1.1;}
.label {font-size: 2.2rem; font-weight: 800; margin: .3rem 0 .2rem; text-transform: capitalize;}
.conf {font-size: 1.15rem; opacity: .85;}
.badge {display:inline-block; padding: .2rem .8rem; border-radius: 999px; font-size: .85rem;
        font-weight: 600; margin-top: .6rem; color: #fff;}
.fruit {background:#f59e0b;} .veg {background:#16a34a;} .low {background:#dc2626;}
.row {display:flex; align-items:center; gap:.6rem; margin: .5rem 0 .1rem; font-weight: 600;
      text-transform: capitalize;}
.row .pct {margin-left: auto; opacity: .75;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🥗 Fruits & Vegetables Classifier</h1>'
    "<p>Upload a photo and the AI will tell you what it is</p></div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(model, img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    # No /255 here: the model's first layer is Rescaling(1/255) already.
    x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    logits = model.predict(x, verbose=0)[0]
    # Last Dense layer is linear -> logits, so apply softmax.
    e = np.exp(logits - logits.max())
    return e / e.sum()


def reference_image(name: str):
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = REF_DIR / f"{name}.{ext}"
        if p.exists():
            return p
    return None


model = load_model()
assert model.output_shape[-1] == len(CLASS_NAMES), (
    f"Model outputs {model.output_shape[-1]} classes but CLASS_NAMES has {len(CLASS_NAMES)}"
)

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if file:
    image = Image.open(file)
    with st.spinner("Analyzing..."):
        probs = predict(model, image)

    top3 = np.argsort(probs)[::-1][:3]
    best = CLASS_NAMES[top3[0]]
    conf = float(probs[top3[0]])

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("📷 Your image")
        st.image(image, width="stretch")

    with right:
        st.subheader("🎯 Prediction")
        ref = reference_image(best)
        if ref:
            st.image(str(ref), width=260)
        else:
            st.markdown(f'<div class="card"><div class="emoji">{CLASSES[best]}</div></div>',
                        unsafe_allow_html=True)

        if conf < 0.5:
            badge = '<span class="badge low">Low confidence</span>'
        elif best in FRUITS:
            badge = '<span class="badge fruit">Fruit</span>'
        else:
            badge = '<span class="badge veg">Vegetable</span>'

        st.markdown(
            f'<div class="card" style="margin-top:.8rem"><div class="label">{best}</div>'
            f'<div class="conf">{conf:.1%} confident</div>{badge}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Top 3 matches")
        for i in top3:
            name = CLASS_NAMES[i]
            st.markdown(
                f'<div class="row"><span>{CLASSES[name]}</span><span>{name}</span>'
                f'<span class="pct">{probs[i]:.1%}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(float(probs[i]))
else:
    st.info("👆 Upload a fruit or vegetable image to get started.")
