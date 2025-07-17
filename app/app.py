import streamlit as st
from PIL import Image
import base64
import torch
import os
from io import BytesIO
import torchvision.transforms as transforms

# Load your full model directly
@st.cache_resource
def load_model():
    model_filename = "efficientnet_fashion_model_full.pth"
    model_path = os.path.abspath(os.path.join("app", model_filename))
    print("Model path:", model_path)

    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model


# Replace with your own labels
@st.cache_data
def load_labels():
    return ["sporty", "vintage", "formal", "casual", "streetwear"]

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("👗 Fashion Style Classifier")
st.write("Upload a fashion image to classify it using EfficientNet.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # Convert image to base64
    img_base64 = image_to_base64(image)

    # Display styled image using HTML
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_base64}"
                 style="border-radius: 15px;
                        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
                        width: 50%;
                        max-width: 500px;" />
            <p style="color: #555; font-size: 16px;">Uploaded Image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Classifying...")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    model = load_model()
    labels = load_labels()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_label = labels[top_catid]

    st.success(f"**Prediction:** {predicted_label} ({top_prob.item():.2%} confidence)")
