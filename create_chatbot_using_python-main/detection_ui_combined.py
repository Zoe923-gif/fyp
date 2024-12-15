import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from DCGAN_ResNet import DCGANTrainer  # Your model class for DCGAN-based defect detection
from hyperbolic_mscnn import HyperbolicMSCNN, manifold  # Hyperbolic MSCNN-related classes
from hypll.tensors import TangentTensor
from SegmentationCrop import process_and_crop_segmented_image
from deeplabv3_plus_ import DeepLabV3Plus
import pdfkit  # For generating PDFs
import time
from io import BytesIO
import pdfkit
import base64

# Add the path to wkhtmltopdf
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

def save_as_pdf(image_name, prediction_result, model_choice, image_bytes):
    """
    Generates a PDF report containing the image, prediction result, and model information.

    Args:
        image_name (str): Name of the uploaded or captured image.
        prediction_result (str): The predicted defect classification ("Defect" or "No Defect").
        model_choice (str): The selected defect detection model ("DCGAN ResNet" or "Hyperbolic MSCNN").
        image_bytes (bytes): Bytes representing the image data.
    """
    # Encode the image in Base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create HTML content with the Base64-encoded image
    html_content = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Defect Detection Report</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
          }}
          h1 {{
            text-align: center;
          }}
          .image-container {{
            text-align: center;
            margin-bottom: 20px;
          }}
          .info-table {{
            border-collapse: collapse;
            width: 100%;
          }}
          .info-table th, .info-table td {{
            padding: 10px;
            border: 1px solid #ddd;
          }}
        </style>
      </head>
      <body>
        <h1>Defect Detection Report</h1>
        <div class="image-container">
          <img src="data:image/png;base64,{base64_image}" alt="Input Image" width="500">
        </div>
        <table class="info-table">
          <tr>
            <th>Image</th>
            <td>{image_name}</td>
          </tr>
          <tr>
            <th>Prediction Result</th>
            <td>{prediction_result}</td>
          </tr>
          <tr>
            <th>Defect Detection Model</th>
            <td>{model_choice}</td>
          </tr>
        </table>
      </body>
    </html>
    """

    # Generate the PDF
    try:
        pdfkit.from_string(html_content, "report.pdf", configuration=config)
        st.success("PDF report generated successfully!")
    except Exception as e:
        st.error(f"Error creating PDF: {e}")

# Paths to model checkpoints
seg_model_path = "C:/Users/zoezh/fyp-2/create_chatbot_using_python-main/create_chatbot_using_python-main/segmentation_model.pth"
dcgan_defect_model_path = 'C:/Users/zoezh/fyp-2/create_chatbot_using_python-main/create_chatbot_using_python-main/dcganResnet_discriminator_6b.pth'
hyperbolic_defect_model_path = 'C:/Users/zoezh/fyp-2/create_chatbot_using_python-main/create_chatbot_using_python-main/hyperbolic_mscnn_gc.pth'

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the segmentation model
segmentation_model = DeepLabV3Plus(num_classes=9)
segmentation_model.load_state_dict(torch.load(seg_model_path, map_location=device))
segmentation_model.to(device)
segmentation_model.eval()

# Load DCGAN Discriminator
dcgan_discriminator = DCGANTrainer.load_model(DCGANTrainer.Discriminator, model_path=dcgan_defect_model_path)

# Load Hyperbolic MSCNN
hyperbolic_mscnn = HyperbolicMSCNN(input_channels=3, num_classes=2)
hyperbolic_mscnn.load_state_dict(torch.load(hyperbolic_defect_model_path, map_location=device))
hyperbolic_mscnn.to(device)
hyperbolic_mscnn.eval()

# Define image transformations
transform_inference = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function for DCGAN
def predict_defect_dcgan(image, model):
    image_tensor = transform_inference(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[1]  # Extract logits if the output is a tuple
        prediction = torch.sigmoid(output).item()  # Assuming binary classification
    return prediction

# Prediction function for Hyperbolic MSCNN
def predict_defect_hyperbolic(image, model):
    image_tensor = transform_inference(image).unsqueeze(0).to(device)
    tangents = TangentTensor(data=image_tensor, man_dim=1, manifold=model.manifold)
    manifold_image = manifold.expmap(tangents)
    with torch.no_grad():
        output = model(manifold_image)
        output_tensor = output.tensor
        if output_tensor.shape[1] == 2:
            prediction = torch.sigmoid(output_tensor[:, 1]).item()
        else:
            prediction = torch.sigmoid(output_tensor).max(1).values.item()
    return prediction

# Streamlit UI
#BANNERRR
st.image("C:/Users/zoezh/fyp-2/create_chatbot_using_python-main/create_chatbot_using_python-main/Banner.png")
st.title("Definuity: Defining Precision, Detecting Perfection")

# Layout using columns
col1, col2 = st.columns(2)

# Model Selection and Image Input Method
with col1:
    model_choice = st.radio("Select defect detection model:", ("DCGAN ResNet", "Hyperbolic MSCNN"))

with col2:
    input_choice = st.radio("Select the image input method:", ("Upload Image", "Use Camera"))

# # Handle input
input_image = None
if input_choice == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_image:
        input_image = Image.open(uploaded_image).convert("RGB")
        # Convert the image to bytes
        image_bytes = input_image.tobytes()
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        input_image = Image.open(BytesIO(camera_image.getvalue())).convert("RGB")  # Handle BytesIO for camera input
        image_bytes = camera_image.getvalue()

# Check if session state keys exist; initialize if not
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store processing messages

# Main layout: Input Image on the left, Processing Steps on the right
if input_image:
    col_left, col_right = st.columns([1, 2])  # Adjust column width ratio

    with col_left:
        # Display input image
        st.image(input_image, caption="Your Input Image", use_column_width=True)

    with col_right:
        try:
            # Step 1: Process the image using the segmentation model and crop the segmented area
            if st.session_state.prediction_result is None:
                # Clear messages for fresh computation
                st.session_state.messages = []
                st.session_state.messages.append("DeepLabV3+ processing segmentation...")
                st.write("DeepLabV3+ processing segmentation...")

                cropped_image = process_and_crop_segmented_image(
                    model=segmentation_model,
                    image=input_image,
                    device=device,
                    target_size=(256, 256),
                    mask_color=(255, 255, 255),
                    exclude_classes=[2, 4, 6, 7]
                )
                st.session_state.messages.append("DeepLabV3+ done segmentation.")
                st.write("DeepLabV3+ done segmentation.")

                # Step 2: Predict defect based on the selected model
                st.session_state.messages.append(f"{model_choice} is predicting defect...")
                st.write(f"{model_choice} is predicting defect...")
                bar = st.progress(0)

                if model_choice == "DCGAN ResNet":
                    prediction = predict_defect_dcgan(cropped_image, dcgan_discriminator)
                else:
                    prediction = predict_defect_hyperbolic(cropped_image, hyperbolic_mscnn)

                time.sleep(3)
                bar.progress(100)
                st.session_state.messages.append("Prediction done successfully.")
                st.success("Prediction done successfully.")

                # Save the prediction result to session state
                st.session_state.prediction_result = "Defect" if prediction > 0.6 else "No Defect"
            else:
                # Replay messages from session state
                for msg in st.session_state.messages:
                    st.write(msg)

            # Step 3: Display result
            result = st.session_state.prediction_result
            st.info(f"Your Clothes Prediction Result is: {result}")

            # Save as PDF and Download buttons
            col_buttons = st.columns([1, 1])
            with col_buttons[0]:
                if st.button("Save Report as PDF"):
                    save_as_pdf(uploaded_image.name if uploaded_image else "captured_image.jpg", result, model_choice, image_bytes)
                    st.session_state.report_ready = True  # Mark report as ready

            with col_buttons[1]:
                if st.session_state.report_ready:
                    with open("report.pdf", "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                    st.download_button(label="Download PDF Report",
                                       data=PDFbyte,
                                       file_name="defect_report.pdf",
                                       mime="application/octet-stream")

        except Exception as e:
            st.error(f"Error processing the image: {e}")