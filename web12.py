import os
import sys

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
import io

# Import rembg dengan fallback
REMBG_AVAILABLE = False
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

st.set_page_config(page_title="Matrix Image Transformation", layout="wide")

# Fungsi untuk menambahkan background video
def add_bg_video(video_file):
    try:
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode()
        
        video_css = f"""
        <style>
        pre, code {{
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            font-weight: bold;
            backdrop-filter: blur(10px);
        }}
        
        /* Font Times New Roman */
        .main .block-container,
        .main .block-container * {{
            font-family: 'Times New Roman', Times, serif !important;
        }}
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {{
            font-family: 'Times New Roman', Times, serif !important;
        }}

        /* Main content styling */
        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 2rem;
            backdrop-filter: blur(5px);
        }}
        
        /* ✅ JUDUL BESAR TENGAH - TARGET STREAMLIT TITLE */
        .main h1,
        .main h2,
        .main h3 {{
            text-align: center !important;
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
            display: block !important;
        }}
        
        /* Target spesifik untuk st.title dan st.header */
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {{
            text-align: center !important;
        }}
        
        /* ✅ SEMUA KONTEN LAIN TETAP KIRI */
        .main p,
        .main div:not([data-testid="stMarkdownContainer"]),
        .main ul,
        .main ol,
        .main li {{
            text-align: left !important;
        }}
        
        .stMarkdown p,
        .stMarkdown ul,
        .stMarkdown ol,
        .stMarkdown li {{
            text-align: left !important;
        }}
        
        #video-background {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            object-fit: cover;
            opacity: 1;
            filter: blur(6.5px);  
            transform: scale(1.1);
        }}
        
        .stApp {{
            background: transparent;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        [data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}
        
        /* Main content styling */
        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 2rem;
            backdrop-filter: blur(5px);
            border: 2px solid rgba(255, 255, 255, 0.2);
        }}
        
        h1, h2, h3 {{
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}
        
        /* ✅ KONTEN UTAMA SAJA → PUTIH */
        section[data-testid="stMain"] p,
        section[data-testid="stMain"] span,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] li,
        section[data-testid="stMain"] h1,
        section[data-testid="stMain"] h2,
        section[data-testid="stMain"] h3 {{
            color: #ffffff !important;
        }}
        
        header[data-testid="stHeader"] {{
            background-color: rgba(255, 255, 255, 0.125) !important;
            backdrop-filter: blur(10px);
        }}
        
        header[data-testid="stHeader"] * {{
            color: #ffffff !important;
        }}

        /* KHUSUS menu popup titik tiga */
        div[data-baseweb="menu"] {{
            background-color: rgba(255, 255, 255, 0.95) !important;
        }}

        div[data-baseweb="menu"] *,
        div[data-baseweb="menu"] li,
        div[data-baseweb="menu"] span,
        div[data-baseweb="menu"] button {{ 
            color: #000000 !important;
        }}
        
        .member-card {{
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.2);
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px 20px;
            color: white !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: rgba(255, 255, 255, 0.3);
        }}

        /* =============================== */
        /* ✅ FILE UPLOADER TEXT → HITAM */
        /* =============================== */
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] label {{
            color: #000000 !important;
            backdrop-filter: blur(10px);
            border-radius: 10px;
        }}

        /* KHUSUS SELECTBOX SIDEBAR */
        [data-testid="stSidebar"] div[data-baseweb="select"] * {{
            color: #000000 !important;
        }}

        /* Kotak selectbox */
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 10px !important;
            border: 2px solid rgba(255, 255, 255, 0.6) !important;
        }}

        /* ===== FIX TOTAL TOMBOL DOWNLOAD (ANTI PUTIH) ===== */
        div[data-testid="stDownloadButton"] {{
            display: flex !important;
            justify-content: center !important;
            margin-top: 1.5rem !important;
        }}

        /* tombol bisa berupa <a> atau <button> tergantung versi Streamlit */
        div[data-testid="stDownloadButton"] a,
        div[data-testid="stDownloadButton"] button {{
            background: linear-gradient(135deg, #ffffff, #eaeaea) !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            padding: 0.7rem 1.6rem !important;
            border: none !important;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.25) !important;
            transition: all 0.25s ease !important;
            text-align: center !important;
        }}

        /* PAKSA SEMUA TEKS DI DALAM TOMBOL JADI HITAM */
        div[data-testid="stDownloadButton"] *,
        div[data-testid="stDownloadButton"] a *,
        div[data-testid="stDownloadButton"] button * {{
            color: #000000 !important;
            fill: #000000 !important;
        }}

        /* hover */
        div[data-testid="stDownloadButton"] a:hover,
        div[data-testid="stDownloadButton"] button:hover {{
            transform: translateY(-3px);
            box-shadow: 0px 12px 28px rgba(0, 0, 0, 0.35) !important;
            background: linear-gradient(135deg, #ffffff, #f5f5f5) !important;
        }}

        /* klik */
        div[data-testid="stDownloadButton"] a:active,
        div[data-testid="stDownloadButton"] button:active {{
            transform: scale(0.97);
        }}

        </style>
        
        <video id="video-background" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """
        st.markdown(video_css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

add_bg_video("bganm.mp4")

# ========================================
# TRANSFORMATION FUNCTIONS
# ========================================

def apply_translation(image, tx, ty):
    """Apply translation using transformation matrix"""
    height, width = image.shape[:2]
    T = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    translated = cv2.warpAffine(image, T, (width, height))
    return translated, T

def apply_scaling(image, sx, sy):
    """Apply scaling using transformation matrix"""
    height, width = image.shape[:2]
    T = np.float32([[sx, 0, 0],
                    [0, sy, 0]])
    new_width = int(width * sx)
    new_height = int(height * sy)
    scaled = cv2.warpAffine(image, T, (new_width, new_height))
    return scaled, T

def apply_rotation(image, angle):
    """Apply rotation using transformation matrix"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rad = np.radians(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)
    
    T = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos_abs = abs(cos_val)
    sin_abs = abs(sin_val)
    new_width = int(height * sin_abs + width * cos_abs)
    new_height = int(height * cos_abs + width * sin_abs)
    
    T[0, 2] += (new_width / 2) - center[0]
    T[1, 2] += (new_height / 2) - center[1]
    
    rotated = cv2.warpAffine(image, T, (new_width, new_height))
    return rotated, T

def apply_shearing(image, shx, shy):
    """Apply shearing using transformation matrix"""
    height, width = image.shape[:2]
    T = np.float32([[1, shx, 0],
                    [shy, 1, 0]])
    
    new_width = int(width + abs(shx * height))
    new_height = int(height + abs(shy * width))
    
    sheared = cv2.warpAffine(image, T, (new_width, new_height))
    return sheared, T

def apply_reflection(image, axis):
    """Apply reflection using transformation matrix"""
    height, width = image.shape[:2]
    
    if axis == "X-axis":
        T = np.float32([[1, 0, 0],
                        [0, -1, height]])
    elif axis == "Y-axis":
        T = np.float32([[-1, 0, width],
                        [0, 1, 0]])
    else:  # Origin
        T = np.float32([[-1, 0, width],
                        [0, -1, height]])
    
    reflected = cv2.warpAffine(image, T, (width, height))
    return reflected, T

def manual_convolution(image, kernel):
    """Apply convolution manually using kernel"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    
    padded = np.pad(gray, pad, mode='edge')
    output = np.zeros_like(gray, dtype=np.float32)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    return output

def apply_blur(image):
    """Apply blur using custom convolution kernel"""
    kernel = np.ones((5, 5), dtype=np.float32) / 25
    return manual_convolution(image, kernel), kernel

def apply_sharpen(image):
    """Apply sharpen using custom convolution kernel"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return manual_convolution(image, kernel), kernel

def apply_background_removal(image, method="AI-Powered (rembg)"):
    """Apply background removal - AI powered for focused objects"""
    
    if method == "AI-Powered (rembg)":
        if not REMBG_AVAILABLE:
            st.error("❌ Library rembg tidak tersedia!")
            st.info("💡 Tambahkan 'rembg>=2.0.50' ke requirements.txt")
            return image
            
        try:
            # Convert numpy → PIL
            pil_image = Image.fromarray(image)
            
            # AI background removal
            with st.spinner("The AI is working..."):
                output = remove(pil_image)
            
            # Convert PIL → numpy
            result = np.array(output)
            
            # Handle transparency → white background
            if result.shape[2] == 4:
                rgb = result[:, :, :3]
                alpha = result[:, :, 3:4] / 255.0
                white_bg = np.ones_like(rgb) * 255
                result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return image
    
    elif method == "Color Thresholding":
        # Untuk background warna solid
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array([80, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(image, image, mask=mask_inv)
        white_bg = np.ones_like(image) * 255
        result = np.where(mask[:,:,np.newaxis] == 255, white_bg, result)
        return result
    
    elif method == "GrabCut":
        # Untuk semi-manual selection
        img_copy = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        height, width = image.shape[:2]
        margin = int(min(width, height) * 0.15)
        rect = (margin, margin, width - margin, height - margin)
        
        try:
            cv2.grabCut(img_copy, mask, rect, bgd_model, fgd_model, 
                       5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = image * mask2[:, :, np.newaxis]
            white_bg = np.ones_like(image) * 255
            result = np.where(mask2[:,:,np.newaxis] == 0, white_bg, result)
            return result
        except:
            st.warning("⚠️ GrabCut failed")
            return image
    
    return image

# ========================================
# MAIN APPLICATION
# ========================================

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Homepage", "Image Processing Tools", "Creator Profile"])

# ========================================
# PAGE 1: HOME & THEORY
# ========================================
if page == "Homepage":
    st.title("Matrix Transformations in Image Processing")
    st.write("### Complete Streamlit Application for Image Transformation Using Matrix Operations")
    st.markdown("""
    ---
    ## About This Application
                
    This web application demonstrates the power of **matrix operations** and **convolution** in digital image processing. 
    Through interactive tools, you can apply various transformations to images and see the mathematical principles in action.

    **What You Can Do:**
    - **Geometric Transformations**: Translation, Scaling, Rotation, Shearing, Reflection
    - **Image Filtering**: Blur (Smoothing) and Sharpen filters using convolution
    - **Background Removal**: Advanced segmentation techniques (Bonus Feature)

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔢 Matrix Transformations
        
        Geometric transformations use **transformation matrices** to modify image coordinates:
        
        **1. Translation** - Move image
        ```
        [1  0  tx]
        [0  1  ty]
        [0  0  1 ]
        ```
        
        **2. Scaling** - Resize image
        ```
        [sx  0]
        [0  sy]
        ```
        
        **3. Rotation** - Rotate image
        ```
        [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)]
        ```
        
        **4. Shearing** - Skew image
        ```
        [1   shx]
        [shy  1 ]
        ```
        
        **5. Reflection** - Mirror image
        ```
        X-axis: [1   0]    Y-axis: [-1  0]
                [0  -1]            [ 0  1]
        ```
        """)

    with col2:
        st.markdown("""
        ### 🎯 Convolution Operations
        
        Convolution applies a **kernel** (small matrix) to each pixel to create effects:
        
        **1. Blur (Smoothing Filter)**
        ```
        1/25 × [1  1  1  1  1]
               [1  1  1  1  1]
               [1  1  1  1  1]
               [1  1  1  1  1]
               [1  1  1  1  1]
        ```
        Averages neighboring pixels to reduce noise and detail.
        
        **2. Sharpen (High-pass Filter)**
        ```
        [ 0  -1   0]
        [-1   5  -1]
        [ 0  -1   0]
        ```
        Enhances edges and details by amplifying differences.
        
        **How Convolution Works:**
        1. Place kernel over each pixel
        2. Multiply overlapping values
        3. Sum the results
        4. Place sum in output image
        """)

    st.markdown("""
    ---
    ## How to Use This App

    1. Go to **"Image Processing Tools"** from sidebar
    2. Upload your image (JPG, PNG, JPEG)
    3. Choose transformation type from sidebar
    4. Adjust parameters using sliders
    5. View results and download processed image

    ---
    """)

# ========================================
# PAGE 2: IMAGE PROCESSING TOOLS
# ========================================
elif page == "Image Processing Tools":
    st.title("Image Processing Tools")
    st.write("Upload an image and apply various transformations")
    
    # Upload Image in main area
    st.write("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Sidebar - Choose Transformation
        st.sidebar.header("Choose Transformation")
        transformation = st.sidebar.selectbox(
            "Select Type:",
            ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", 
             "Blur", "Sharpen", "Background Removal"]
        )
        
        # Apply transformation based on selection
        transformed_image = None
        matrix = None
        kernel = None
        
        if transformation == "Translation":
            st.sidebar.subheader("Translation Parameters")
            tx = st.sidebar.slider("Translate X (pixels):", -200, 200, 50)
            ty = st.sidebar.slider("Translate Y (pixels):", -200, 200, 50)
            transformed_image, matrix = apply_translation(original_image, tx, ty)
            
        elif transformation == "Scaling":
            st.sidebar.subheader("Scaling Parameters")
            sx = st.sidebar.slider("Scale X:", 0.1, 3.0, 1.5)
            sy = st.sidebar.slider("Scale Y:", 0.1, 3.0, 1.5)
            transformed_image, matrix = apply_scaling(original_image, sx, sy)
            
        elif transformation == "Rotation":
            st.sidebar.subheader("Rotation Parameters")
            angle = st.sidebar.slider("Angle (degrees):", 0, 360, 45)
            transformed_image, matrix = apply_rotation(original_image, angle)
            
        elif transformation == "Shearing":
            st.sidebar.subheader("Shearing Parameters")
            shx = st.sidebar.slider("Shear X:", -1.0, 1.0, 0.5)
            shy = st.sidebar.slider("Shear Y:", -1.0, 1.0, 0.0)
            transformed_image, matrix = apply_shearing(original_image, shx, shy)
            
        elif transformation == "Reflection":
            st.sidebar.subheader("Reflection Parameters")
            axis = st.sidebar.selectbox("Reflect over:", ["X-axis", "Y-axis", "Origin"])
            transformed_image, matrix = apply_reflection(original_image, axis)
            
        elif transformation == "Blur":
            st.sidebar.subheader("Blur Filter")
            st.sidebar.write("Using 5x5 averaging kernel")
            transformed_image, kernel = apply_blur(original_image)
            
        elif transformation == "Sharpen":
            st.sidebar.subheader("Sharpen Filter")
            st.sidebar.write("Using edge enhancement kernel")
            transformed_image, kernel = apply_sharpen(original_image)
            
        elif transformation == "Background Removal":
            st.sidebar.subheader("Background Removal")
            st.sidebar.write("**Best for: Humans, Animals, Objects**")
            method = st.sidebar.selectbox("Method:", 
                ["AI-Powered (rembg)", "GrabCut", "Color Thresholding"])
    
            if method == "AI-Powered":
                st.sidebar.info("Using AI model\n\nWorks best for:\n- Portraits\n- Animals\n- Products")
    
            transformed_image = apply_background_removal(original_image, method)

        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.subheader("✨ Transformed Image")
            if transformed_image is not None:
                st.image(transformed_image, use_container_width=True)
        
        # Display Matrix/Kernel
        if matrix is not None:
            st.subheader("🔢 Transformation Matrix")
            st.code(str(matrix))
        
        if kernel is not None:
            st.subheader("🎯 Convolution Kernel")
            st.code(str(kernel))
        
        # Download button
        if transformed_image is not None:
            result_pil = Image.fromarray(transformed_image)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="💾 Download Transformed Image",
                data=byte_im,
                file_name=f"transformed_{transformation.lower().replace(' ', '_')}.png",
                mime="image/png"
            )

    else:
        st.info("**👆 Please upload an image to begin!**")

        st.write("**Supported transformations:**")
        st.write("- Translation, Scaling, Rotation, Shearing, Reflection")
        st.write("- Blur and Sharpen filters")
        st.write("- Background Removal")

# ========================================
# PAGE 3: CREATOR PROFILE
# ========================================
elif page == "Creator Profile":
    st.title("Creator Profile")
    st.write("### Meet the developers behind this project")
    
    st.markdown("---")
    
    # Creator Profile information
    creator_profile = [
        {
            "name": "Fayruz Novarendi",
            "Student ID" : "004202400053",
            "Class" : "Industrial Engineering Class 1 Group 8",
            "Course" : "Algebra Linear",
            "role": "Project Leader & Backend Developer",
            "contribution": "Creating the website",
            "photo": "author.jpg"
        },
    ]
    
    # Display team members
    for i, member in enumerate(creator_profile):
    # Buat 3 kolom, kosongkan kolom 1 dan 3
        left, center, right = st.columns([3, 2, 3])
    
        with center:
            # Foto saja dulu
            try:
                if os.path.exists(member["photo"]):
                    st.image(member["photo"], width=250, use_container_width=False)
                else:
                    st.image(f"https://via.placeholder.com/250x250.png?text={member['name'].replace(' ', '+')}", 
                            width=250, use_container_width=False)
            except:
                st.image(f"https://via.placeholder.com/250x250.png?text={member['name'].replace(' ', '+')}", 
                        width=250, use_container_width=False)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ✅ Info member dalam container penuh
        st.markdown(f"""
        <div class="member-card">
            <h3 style="color: #ffffff !important; text-align: center;">{member["name"]}</h3>
            <p style="color: #FFD700 !important;"><strong>Student ID:</strong> {member["Student ID"]}</p>
            <p style="color: #FFD700 !important;"><strong>Class:</strong> {member["Class"]}</p>
            <p style="color: #FFD700 !important;"><strong>Course:</strong> {member["Course"]}</p>
            <p style="color: #FFD700 !important;"><strong>Role:</strong> {member["role"]}</p>
            <p style="color: #ffffff !important;"><strong>Contribution:</strong> {member["contribution"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if i < len(creator_profile) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How the App Works
    st.header("🔧 How This Application Works")
    
    st.markdown("""
    ### Technical Implementation

    **Matrix-Based Geometric Transformations:**
    - All geometric transformations use custom transformation matrices
    - OpenCV's `warpAffine` is used for applying the matrices to images
    - Each transformation matrix is displayed to show the mathematical operation

    **Convolution-Based Filtering:**
    - Blur and Sharpen filters use custom convolution kernels
    - Manual convolution implementation (not using built-in OpenCV blur functions)
    - Kernels are applied pixel-by-pixel across the image

    **Image Processing Pipeline:**
    1. User uploads image in main area
    2. Selects transformation type from sidebar
    3. Adjusts parameters using sidebar sliders
    4. Transformation matrix/kernel is generated
    5. Operation is applied to the image
    6. Results are displayed side-by-side with the original
    7. Processed image can be downloaded

    ### Technologies Used
    - **Streamlit** - Web application framework
    - **NumPy** - Matrix operations and numerical computing
    - **OpenCV** - Image manipulation and computer vision
    - **PIL (Pillow)** - Image file handling
    - **Python** - Core programming language

    ### Key Features
    ✅ All 5 geometric transformations implemented  
    ✅ Custom convolution kernels for Blur and Sharpen  
    ✅ Background removal (bonus feature)  
    ✅ Interactive parameter adjustment  
    ✅ Real-time preview  
    ✅ Download transformed images  
    ✅ Matrix/kernel visualization 
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #ffffff; padding: 20px;'>
        <h4>Thank you for using my website! 🎉</h4>
        <h5>Matrix Transformations & Image Processing Course Project</h5>
    </div>
    """, unsafe_allow_html=True)