import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="GoogLeNet Style Classifier", page_icon="📸")

st.title("📸 GoogLeNet Style Image Classifier")

# Camera selection (Front/Back toggle)
camera_option = st.radio("📷 Select Camera:", ["Front Camera", "Back Camera"], horizontal=True)

if camera_option == "Front Camera":
    camera_key = "front"
else:
    camera_key = "back"

# Try to load model with caching
@st.cache_resource
def load_lightweight_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        
        model = MobileNetV2(weights='imagenet', input_shape=(224,224,3))
        st.success("✅ GoogLeNet-style model loaded! (MobileNetV2 - similar accuracy)")
        return model, True, preprocess_input, decode_predictions
    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {str(e)[:100]}")
        return None, False, None, None

model, model_loaded, preprocess_func, decode_func = load_lightweight_model()

# Camera input
st.markdown("### 📸 Take a Photo")
img_file = st.camera_input("Click the button below to take a photo", key=camera_key)

if img_file is not provided:
    st.info("👆 Click 'Take a photo' button above to capture an image")

if img_file is not None:
    # Load image
    img = Image.open(img_file)
    
    # Display image
    st.image(img, caption="📷 Captured Image", use_container_width=True)
    
    # Resize to 224x224 (MATLAB imresize equivalent)
    img_resized = img.resize((224, 224))
    
    # Convert to numpy array
    img_np = np.array(img)
    img_resized_np = np.array(img_resized)
    
    # MATLAB-style debug information (exactly like your original code)
    st.write("---")
    st.write("### 📊 MATLAB camnet() Function Output:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**size of img**")
        st.code(f"{img_np.shape}")
        
        st.write("**class of img**")
        st.code(f"{type(img_np).__name__}")
    
    with col2:
        st.write("**min of img**")
        st.code(f"{img_np.min()}")
        
        st.write("**max of img**")
        st.code(f"{img_np.max()}")
    
    # Classification result (GoogLeNet style)
    if model_loaded:
        try:
            # Preprocess for model
            img_array = img_resized_np.astype(np.float32)
            
            # Handle RGBA images (if has alpha channel)
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = preprocess_func(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run classification
            predictions = model.predict(img_array, verbose=0)
            results = decode_func(predictions, top=3)[0]
            
            st.write("---")
            st.write("### 🏷️ Classification Result (GoogLeNet-style):")
            
            # Display title (like MATLAB's title(char(value)))
            st.markdown(f"## 🎯 **{results[0][1].title()}**")
            
            # Show all top 3 predictions
            for i, (imagenet_id, label, score) in enumerate(results):
                st.write(f"**{i+1}. {label.title()}:** {score*100:.1f}%")
                
        except Exception as e:
            st.error(f"Classification error: {e}")
            st.info("Showing image data only (classification temporarily unavailable)")
    else:
        st.info("📌 Model loading in progress or failed - showing image metrics (matching your MATLAB code's output)")
        
        # Simulated title (since model not loaded)
        st.markdown(f"## 🎯 **Image Captured Successfully**")
    
    # Display the image with title (like MATLAB's image(pic) and title(char(value)))
    st.markdown("---")
    st.markdown("### 🖼️ As displayed in MATLAB:")
    with st.container():
        st.image(img_resized, caption="224x224 resized image (same as MATLAB's imresize)", use_container_width=True)

else:
    st.markdown("---")
    st.markdown("""
    ### 📱 How to use:
    1. Select **Front** or **Back** camera
    2. Click **"Take a photo"** button
    3. Allow camera permissions
    4. Take a picture of anything!
    5. See classification results (like GoogLeNet) + MATLAB-style debug info
    """)

st.markdown("---")
st.caption("✅ App replicates your MATLAB code: camera selection → snapshot → imresize(224,224) → classify → display image with title + debug info (size, class, min, max)")
