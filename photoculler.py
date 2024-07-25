### Import necessary libraries
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

### Create a Streamlit app title
st.title("Photo Culling Model")

### Load the pre-trained image classification model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

### Define a function to preprocess the input images
def preprocess_images(images):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return [preprocess(image).unsqueeze(0) for image in images]

### Create a file uploader for the user to upload multiple images
uploaded_files = st.file_uploader("Upload multiple images to be culled", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

### Create a button to run the culling model
if st.button("Run Culling Model"):
    ### Check if images have been uploaded
    if uploaded_files is None:
        st.write("Please upload images to run the culling model.")
    else:
        ### Load the uploaded images and preprocess them
        images = [Image.open(file) for file in uploaded_files]
        input_tensors = preprocess_images(images)

        ### Run the images through the culling model
        with torch.no_grad():
            outputs = [model(tensor) for tensor in input_tensors]

        ### Get the predicted classes and confidence scores
        predicted_classes = [torch.max(output.data, 1)[1].item() for output in outputs]
        confidence_scores = [torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() for output, predicted_class in zip(outputs, predicted_classes)]

        ### Display the predicted classes and confidence scores
        for i, (predicted_class, confidence_score) in enumerate(zip(predicted_classes, confidence_scores)):
            st.write(f"Image {i+1}: Predicted Class {predicted_class}, Confidence Score {confidence_score:.2f}")

        ### Display the original images
        st.write("Original Images:")
        for image in images:
            st.image(image, caption=f"Image {images.index(image)+1}")

        ### Display the preprocessed images
        st.write("Preprocessed Images:")
        for tensor in input_tensors:
            st.image(tensor.cpu().numpy().transpose(1, 2, 0), caption=f"Image {input_tensors.index(tensor)+1}")