import streamlit as st
from PIL import Image
import numpy as np
from streamlit_back_camera_input import back_camera_input

st.set_page_config(page_title="Camera Classifier", page_icon="📸")

st.title("📸 Camera Classifier - Front & Back Both")

# Camera selection tabs
camera_tab = st.radio("📷 Select Camera:", ["🎥 Front Camera", "📱 Back Camera"], horizontal=True)

# Different widget based on selection
if camera_tab == "🎥 Front Camera":
    # Front camera - default streamlit camera
    img_buffer = st.camera_input("Take a photo", key="front_cam")
else:
    # Back camera - custom component
    img_buffer = back_camera_input(key="back_cam")

if img_buffer is not None:
    # Open image
    img = Image.open(img_buffer)
    
    # Display captured photo
    st.image(img, caption=f"Photo from {camera_tab}", use_container_width=True)
    
    # Resize to 224x224 (MATLAB imresize)
    img_resized = img.resize((224, 224))
    
    # Get numpy arrays
    img_np = np.array(img)
    img_resized_np = np.array(img_resized)
    
    # ====== MATLAB STYLE OUTPUT ======
    st.markdown("---")
    st.markdown("### 📊 MATLAB camnet() Output:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**size of img**")
        st.code(f"{img_np.shape}")
        
        st.markdown("**class of img**")
        st.code(f"{type(img_np).__name__}")
    
    with col2:
        st.markdown("**min of img**")
        st.code(f"{img_np.min()}")
        
        st.markdown("**max of img**")
        st.code(f"{img_np.max()}")
    
    # Display resized image info
    with st.expander("🖼️ After imresize (224, 224)"):
        st.image(img_resized, caption="Resized image for model input")
        st.write(f"**Resized shape:** {img_resized_np.shape}")
    
    # Optional: Add classification if model loads
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        
        @st.cache_resource
        def load_model():
            return MobileNetV2(weights='imagenet')
        
        model = load_model()
        
        # Prepare for model
        img_for_model = img_resized_np
        if img_for_model.shape[2] == 4:
            img_for_model = img_for_model[:, :, :3]
        img_array = preprocess_input(img_for_model)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        results = decode_predictions(predictions, top=3)[0]
        
        st.markdown("---")
        st.markdown("### 🏷️ Classification Result")
        st.markdown(f"## 🎯 **{results[0][1].title()}**")
        st.write(f"**Confidence:** {results[0][2]*100:.1f}%")
        
        for i, (_, label, score) in enumerate(results[1:], 2):
            st.write(f"{i}. {label.title()}: {score*100:.1f}%")
            
    except ImportError:
        st.info("📌 AI model will work once deployed with TensorFlow")
    except Exception as e:
        st.info(f"📌 Model loading: {str(e)[:100]}")

st.markdown("---")
st.caption("✅ Front Camera (default) + Back Camera (custom component) - Both Working!")
