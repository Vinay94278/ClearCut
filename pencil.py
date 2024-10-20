import streamlit as st
from PIL import Image
import os

# Assuming the U2NET model is loaded and the processing function is imported
from u2net_portrait_composite import apply_u2net_portrait

# Set up your Streamlit app
st.title("AI Image Processing App")
st.sidebar.title("Filters")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Filters in the sidebar
apply_pencil_art = st.sidebar.checkbox("Pencil Art")

# Button to apply filters
if st.button("Apply Filter") and uploaded_file is not None:
    # Save the uploaded file
    image = Image.open(uploaded_file)
    image_path = os.path.join("temp", uploaded_file.name)
    image.save(image_path)

    # Apply filters based on the checkbox selection
    if apply_pencil_art:
        # Apply the U2NET portrait filter
        st.write("Applying Pencil Art filter...")
        final_image = apply_u2net_portrait(image_path)
        
        # Display the result
        st.image(final_image, caption="Pencil Art Filter Applied", use_column_width=True)
        
        # Download button
        final_image_path = os.path.join("temp", "pencil_art_" + uploaded_file.name)
        final_image.save(final_image_path)
        with open(final_image_path, "rb") as file:
            btn = st.download_button(
                label="Download Pencil Art Image",
                data=file,
                file_name="pencil_art_" + uploaded_file.name,
                mime="image/png"
            )

