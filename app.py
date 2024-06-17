import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer
from io import BytesIO
import cv2

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

# Load model once at the start
@st.cache_resource
def load_model():
    return KerasLayer("https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1", trainable=False)

# Main function for Streamlit app
def main():
    st.title("Background Remover using U2-Net")

    np.random.seed(42)
    tf.random.set_seed(42)

    model = load_model()

    # Input image upload section
    st.subheader("Select an image for background removal:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "avif"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Store the original image in session state
        st.session_state['original_image'] = image
        st.session_state['masked_image'] = None
        st.session_state['masked_image_with_bg'] = None

        # Color selection
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

        # Button to remove background
        if st.button('Remove Background'):
            try:
                masked_image = remove_background(model, image)
                masked_image_with_bg = apply_background_color(masked_image, selected_color)

                # Store processed images in session state
                st.session_state['masked_image'] = masked_image
                st.session_state['masked_image_with_bg'] = masked_image_with_bg

            except Exception as e:
                st.error(f"Error processing image: {e}")

    # Check if processed images are available in session state
    if 'masked_image' in st.session_state and 'masked_image_with_bg' in st.session_state:
        masked_image = st.session_state['masked_image']
        masked_image_with_bg = st.session_state['masked_image_with_bg']
        original_image = st.session_state['original_image']

        # Display all three images in a single row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(original_image, caption='Original Image', use_column_width=True)

        with col2:
            if masked_image is not None:
                st.image(masked_image, caption='Removed Background', use_column_width=True)
                # Convert the removed background image to PNG and create download link
                masked_image_pil = Image.fromarray(masked_image, 'RGBA')
                buffered_removed_bg = BytesIO()
                masked_image_pil.save(buffered_removed_bg, format="PNG")
                buffered_removed_bg.seek(0)
                st.download_button(
                    label="Download Removed BG Image",
                    data=buffered_removed_bg,
                    file_name="removed_bg.png",
                    mime="image/png"
                )

        with col3:
            if masked_image_with_bg is not None:
                st.image(masked_image_with_bg, caption='With New Background', use_column_width=True)
                # Convert the new background image to PNG and create download link
                masked_image_with_bg_pil = Image.fromarray(masked_image_with_bg)
                buffered_with_bg = BytesIO()
                masked_image_with_bg_pil.save(buffered_with_bg, format="PNG")
                buffered_with_bg.seek(0)
                st.download_button(
                    label="Download Image with New BG",
                    data=buffered_with_bg,
                    file_name="new_bg.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
