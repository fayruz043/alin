# Matrix Transformation Image Processing

Linear Algebra project implementing geometric transformations and AI-powered background removal.

## Features
- 5 Geometric Transformations
- 2 Convolution Filters  
- AI Background Removal

# Matrix Transformations in Image Processing - Streamlit Web Application

## 📋 Project Overview

This is a multi-page Streamlit web application that demonstrates matrix operations and convolution in digital image processing. The application implements geometric transformations and image filtering using custom matrices and kernels.

## ✨ Features

### Geometric Transformations (Matrix-based)
1. **Translation** - Move image using translation matrix
2. **Scaling** - Resize image using scaling matrix
3. **Rotation** - Rotate image using rotation matrix
4. **Shearing** - Skew image using shearing matrix
5. **Reflection** - Mirror image across X-axis, Y-axis, or origin

### Image Filtering (Convolution-based)
6. **Blur** - Smoothing filter using custom 5x5 averaging kernel
7. **Sharpen** - Edge enhancement using custom high-pass filter kernel

### Bonus Feature
8. **Background Removal** - Using Color Thresholding or GrabCut segmentation

## 📁 Project Structure

```
project/
├── Home.py                          # Main page (Landing page)
├── pages/
│   ├── 1_Image_Processing_Tools.py  # Image transformation tools
│   └── 2_Team_Members.py            # Team member information
├── bganm.mp4                        # Background video (optional)
├── member1.jpg                      # Team member photos
├── member2.jpg
├── member3.jpg
├── member4.jpg
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Libraries
- streamlit
- numpy
- opencv-python
- pillow
- matplotlib

### 3. Run the Application

```bash
streamlit run Home.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📝 How to Use

1. **Navigate** using the sidebar menu
2. **Upload** your image (JPG, PNG, JPEG format)
3. **Select** transformation type
4. **Adjust** parameters using sliders
5. **View** original vs transformed image
6. **Download** the processed image

## 🎨 Customization

### Adding Team Member Information

Edit `pages/2_Team_Members.py` and modify the `team_members` list:

```python
team_members = [
    {
        "name": "Your Name",
        "role": "Your Role",
        "contribution": "What you did",
        "photo": "your_photo.jpg"
    },
    # Add more members...
]
```

### Adding Team Member Photos

1. Place team member photos in the project root directory
2. Name them as: `member1.jpg`, `member2.jpg`, etc.
3. Update the `photo` field in the `team_members` list

### Adding Background Video (Optional)

1. Place your video file named `bganm.mp4` in the project root directory
2. The video will automatically be used as an animated background
3. If no video file is found, the app will work normally without it

## 🔧 Technical Implementation

### Matrix Transformations
All geometric transformations use 2D transformation matrices:
- Implemented using NumPy arrays
- Applied to images using OpenCV's `warpAffine`
- Matrices are displayed to users for educational purposes

### Convolution Operations
- Custom manual convolution implementation
- Does NOT use built-in OpenCV blur functions
- Kernels are applied pixel-by-pixel

### Code Organization
- Modular function design
- Clear separation of concerns
- Well-documented code

## 📊 Transformation Matrices Reference

### Translation
```
[1  0  tx]
[0  1  ty]
[0  0  1 ]
```

### Scaling
```
[sx  0  0]
[0  sy  0]
```

### Rotation
```
[cos(θ)  -sin(θ)  0]
[sin(θ)   cos(θ)  0]
```

### Shearing
```
[1   shx  0]
[shy  1   0]
```

### Reflection
```
X-axis: [1   0  0]    Y-axis: [-1  0  width]
        [0  -1  h]            [ 0  1  0    ]
```

## 🎯 Convolution Kernels

### Blur (5x5 Averaging)
```
1/25 × [1  1  1  1  1]
       [1  1  1  1  1]
       [1  1  1  1  1]
       [1  1  1  1  1]
       [1  1  1  1  1]
```

### Sharpen
```
[ 0  -1   0]
[-1   5  -1]
[ 0  -1   0]
```

## 📦 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Important Notes for Deployment
- Make sure all file paths are relative
- Include `requirements.txt`
- Test locally before deploying
- Background video is optional (app works without it)

## 🐛 Troubleshooting

### Common Issues

**Issue:** Video background not showing
- **Solution:** Make sure `bganm.mp4` is in the correct location or the app will run without it

**Issue:** Team member photos not displaying
- **Solution:** Place photos in the root directory with correct filenames

**Issue:** Convolution is slow
- **Solution:** Manual convolution is computationally intensive. For large images, it may take a few seconds

**Issue:** Transformation matrix not displaying
- **Solution:** Make sure you've selected a geometric transformation (not blur/sharpen)

## 👥 Team

Update this section with your actual team members!

- **Member 1** - Project Leader
- **Member 2** - Frontend Developer
- **Member 3** - Image Processing Specialist
- **Member 4** - Documentation & Testing

## 📄 License

This project is created for educational purposes as part of a university course assignment.

## 🙏 Acknowledgments

- Course: Matrix Transformations & Image Processing
- Institution: [Your University Name]
- Instructor: [Instructor Name]

---

**Note:** Remember to customize the team member information and add your actual photos before submission!