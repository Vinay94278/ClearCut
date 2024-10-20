import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer
from io import BytesIO
import cv2
from u2net_portrait_composite import apply_u2net_portrait

# Function to remove background using U2-Net model
def remove_background(model, image):
    image_np = np.array(image, dtype="uint8")
    h, w, channel_count = image_np.shape

    if channel_count > 3:
        image_np = image_np[..., :3]

    x = cv2.resize(image_np, (512, 512))  # Resize input image to 512 x 512 x 3
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    probability = model(x)[0].numpy()

    probability = cv2.resize(probability, dsize=(w, h))  # Resize the probability mask from (512, 512) to (h, w)
    probability = np.expand_dims(probability, axis=-1)  # Reshape the probability mask from (h, w) to (h, w, 1)

    alpha_image = np.insert(image_np, 3, 255, axis=2)  # Add an opaque alpha channel to the input image
    PROBABILITY_THRESHOLD = 0.7

    masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0)
    masked_image = masked_image.astype(np.uint8)
    return masked_image

def apply_background_color(masked_image, color):
    # Create a background image with the selected color
    h, w, _ = masked_image.shape
    background = np.full((h, w, 3), color, dtype=np.uint8)

    # Create an alpha mask
    alpha_mask = masked_image[..., 3] / 255.0

    # Blend the masked image with the background color
    for c in range(3):
        masked_image[..., c] = (1.0 - alpha_mask) * background[..., c] + alpha_mask * masked_image[..., c]

    return masked_image[..., :3]

# Function to apply pencil sketch filter
def apply_pencil_sketch(image):
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image_inv = cv2.bitwise_not(image_gray)
    image_smooth = cv2.GaussianBlur(image_inv, (21, 21), sigmaX=0, sigmaY=0)
    pencil_sketch = cv2.divide(image_gray, 255 - image_smooth, scale=256)
    return cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2RGB)

# Function to apply cartoon effect
def apply_cartoon_effect(image):
    image_color = cv2.bilateralFilter(np.array(image), 9, 75, 75)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
    image_edges = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    image_cartoon = cv2.bitwise_and(image_color, image_color, mask=image_edges)
    return image_cartoon

# Load model once at the start
@st.cache_resource
def load_model():
    return KerasLayer("https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1", trainable=False)

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Background Remover", "Apply Filters"])

    model = load_model()

    if option == "Background Remover":
        st.title("Background Remover using U2-Net")

        np.random.seed(42)
        tf.random.set_seed(42)

        # Input image upload section
        st.subheader("Select an image for background removal:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "avif"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state['original_image'] = image

            st.subheader("Select background color:")
            predefined_colors = {
                "Red": [255, 0, 0],
                "Green": [0, 255, 0],
                "Blue": [0, 0, 255],
                "White": [255, 255, 255],
                "Black": [0, 0, 0]
            }
            color_choice = st.selectbox("Choose a predefined color or enter a custom color code:", list(predefined_colors.keys()) + ["Custom"])
            
            if color_choice == "Custom":
                custom_color = st.color_picker("Pick a color:", "#000000")
                selected_color = [int(custom_color[i:i+2], 16) for i in (1, 3, 5)]
            else:
                selected_color = predefined_colors[color_choice]

            if st.button('Remove Background'):
                try:
                    masked_image = remove_background(model, image)
                    masked_image_with_bg = apply_background_color(masked_image, selected_color)

                    st.session_state['masked_image'] = masked_image
                    st.session_state['masked_image_with_bg'] = masked_image_with_bg

                except Exception as e:
                    st.error(f"Error processing image: {e}")

        if 'masked_image' in st.session_state and 'masked_image_with_bg' in st.session_state:
            masked_image = st.session_state['masked_image']
            masked_image_with_bg = st.session_state['masked_image_with_bg']
            original_image = st.session_state['original_image']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(original_image, caption='Original Image', use_column_width=True)

            with col2:
                if masked_image is not None:
                    st.image(masked_image, caption='Removed Background', use_column_width=True)
                    masked_image_pil = Image.fromarray(masked_image, 'RGBA')
                    buffered_removed_bg = BytesIO()
                    masked_image_pil.save(buffered_removed_bg, format="PNG")
                    buffered_removed_bg.seek(0)
                    st.download_button(
                        label="Download Removed BG Image",
                        data=buffered_removed_bg,
                        file_name="removed_bg.png",
                        mime="image/png",
                        key='download_removed_bg'
                    )

            with col3:
                if masked_image_with_bg is not None:
                    st.image(masked_image_with_bg, caption='With New Background', use_column_width=True)
                    masked_image_with_bg_pil = Image.fromarray(masked_image_with_bg)
                    buffered_with_bg = BytesIO()
                    masked_image_with_bg_pil.save(buffered_with_bg, format="PNG")
                    buffered_with_bg.seek(0)
                    st.download_button(
                        label="Download Image with New BG",
                        data=buffered_with_bg,
                        file_name="new_bg.png",
                        mime="image/png",
                        key='download_with_bg'
                    )

    elif option == "Apply Filters":
        st.title("Apply Filters to Image")

        st.subheader("Select an image to apply filters:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "avif"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Checkbox for filters
            apply_pencil = st.checkbox("Pencil Sketch")
            apply_cartoon = st.checkbox("Cartoon Effect")

            if apply_pencil or apply_cartoon:
                col1, col2 = st.columns(2)

                if apply_pencil:
                    pencil_image = apply_pencil_sketch(image)
                    # pencil_image = apply_u2net_portrait(image,sigma=0,alpha=0)
                    with col1:
                        st.image(pencil_image, caption='Pencil Sketch', use_column_width=True)
                        pencil_image_pil = Image.fromarray(pencil_image)
                        print(type(pencil_image))
                        buffered_pencil = BytesIO()
                        pencil_image_pil.save(buffered_pencil, format="PNG")
                        buffered_pencil.seek(0)
                        st.download_button(
                            label="Download Pencil Sketch Image",
                            data=buffered_pencil,
                            file_name="pencil_sketch.png",
                            mime="image/png",
                            key='download_pencil'
                        )

                if apply_cartoon:
                    cartoon_image = apply_cartoon_effect(image)
                    # cartoon_image = apply_u2net_portrait(image,sigma=20,alpha=0.50)
                    with col2:
                        st.image(cartoon_image, caption='Cartoon Effect', use_column_width=True)
                        cartoon_image_pil = Image.fromarray(cartoon_image)
                        buffered_cartoon = BytesIO()
                        cartoon_image_pil.save(buffered_cartoon, format="PNG")
                        buffered_cartoon.seek(0)
                        st.download_button(
                            label="Download Cartoon Effect Image",
                            data=buffered_cartoon,
                            file_name="cartoon_effect.png",
                            mime="image/png",
                            key='download_cartoon'
                        )

if __name__ == "__main__":
    main()