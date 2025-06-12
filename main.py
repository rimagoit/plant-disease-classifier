
import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home", "About", "Diagnosis & Treatment"])


#Main Page
if(app_mode=="Home"):
    # Title and Tagline
    st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h1 style="color:#2e7d32; font-size: 45px; margin-bottom: 10px;">🌿 CropCare AI</h1>
            <h3 style="color: #555; font-weight: normal;">Smart Plant Disease Classifier & Treatment Advisor</h3>
            <p style="font-size: 18px; color: #666;">Classify leaf diseases and get instant treatment advice.</p>
        </div>
    """, unsafe_allow_html=True)

    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)

    st.markdown("""
    <h3 style='text-align: center; color: #4CAF50;'>Welcome to CropCare AI – Your Smart Plant Disease Classifier & Treatment Advisor 🌿🔍</h3>


    Our mission is to assist in the early detection of plant leaf diseases. Upload a clear image of a plant leaf, and our system will analyze it to identify any potential diseases and provide treatment advice. Together, let's protect our crops and promote a healthier harvest!

   ### How It Works
   1. **Upload Image:** Go to the **Diagnosis & Treatment** page and upload an image of a plant leaf showing symptoms.
   2. **Classification:** Our system will analyze the image using a machine learning model to classify the disease type.
   3. **Results:** View the predicted disease class and get treatment recommendations tailored to it.


    ### Why Reference Images are Included:
    We provide reference images alongside the treatment advice for a better understanding of the disease. Here's why:
    
    - **Visual Confirmation:** Helps you visually compare your uploaded image with a reference image of the disease to validate the model's prediction.
    - **Improved Understanding:** Many users may not be familiar with disease names. A visual example makes it easier to understand what the disease looks like.
    - **Professional Appearance:** It improves the overall UX/UI of your app, making it look polished, educational, and reliable.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease classification.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Diagnosis & Treatment** page in the sidebar to upload an image and experience the power of Smart Plant Disease Classifier & Treatment Advisor!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """, unsafe_allow_html=True)


#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.

    A new directory containing 33 test images is created later for prediction purposes.

    **Dataset Structure:**
    1. Train (70,295 images)
    2. Test (33 images)
    3. Validation (17,572 images)

    #### Technologies Used
    - **Machine Learning Framework:** TensorFlow / Keras
    - **Interface:** Streamlit
    - **Image Preprocessing:** TensorFlow ImageDataGenerator
    - **Deployment:** Local deployment (can be extended to cloud platforms)

    """)

#Prediction Page
elif(app_mode=="Diagnosis & Treatment"):
    st.header("Diagnosis & Treatment")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if st.button("Predict"):
      with st.spinner("Analyzing image, please wait..."):
        result_index = model_prediction(test_image)
      st.success("Analysis complete!")
      st.write("Our Prediction")

        #Reading Labels
      class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
                    
      treatment_dict = {
    'Apple___Apple_scab': """
    Remove and destroy infected leaves. Apply fungicides like captan or myclobutanil during early fruit development.<br><br>
    <a href="https://extension.umn.edu/plant-diseases/apple-scab" target="_blank">Read More</a><br>
    <img src="https://www.gardeningknowhow.com/wp-content/uploads/2021/08/apple-scab.jpg" width="300">
    """,
        
    'Apple___Black_rot': """
    Prune and remove infected branches. Use fungicides like captan. Sanitize pruning tools.<br><br>
    <a href="https://extension.umn.edu/plant-diseases/black-rot-apples" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/black-rot-apple.jpg" width="300">
    """,

    'Apple___Cedar_apple_rust': """
    Remove nearby cedar trees. Apply fungicides like mancozeb in early spring.<br><br>
    <a href="https://extension.umn.edu/plant-diseases/cedar-apple-rust" target="_blank">Read More</a><br>
    <img src="https://www.gardeningknowhow.com/wp-content/uploads/2021/08/cedar-rust.jpg" width="300">
    """,

    'Apple___healthy': """
    No disease detected. Keep the plant well-pruned and monitor regularly.<br><br>
    <img src="https://www.gardeningknowhow.com/wp-content/uploads/2021/08/healthy-apple.jpg" width="300">
    """,

    'Blueberry___healthy': """
    Healthy plant. Maintain soil pH, ensure proper watering and apply mulch.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/Blueberries.jpg" width="300">
    """,

    'Cherry_(including_sour)___Powdery_mildew': """
    Apply sulfur-based fungicides. Improve air circulation and prune affected branches.<br><br>
    <a href="https://extension.umn.edu/plant-diseases/powdery-mildew-trees-and-shrubs" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/powdery-mildew-cherry.jpg" width="300">
    """,

    'Cherry_(including_sour)___healthy': """
    Cherry tree is healthy. Ensure regular pruning and pest monitoring.<br><br>
    <img src="https://cdn.britannica.com/16/126216-050-73BE57CB/Sour-cherries.jpg" width="300">
    """,

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': """
    Use resistant hybrids and apply fungicides like strobilurins at VT stage.<br><br>
    <a href="https://cropprotectionnetwork.org/resources/articles/diseases/corn-gray-leaf-spot" target="_blank">Read More</a><br>
    <img src="https://cropwatch.unl.edu/images/gray-leaf-spot-corn.jpg" width="300">
    """,

    'Corn_(maize)___Common_rust_': """
    Use resistant varieties and apply fungicides early if rust appears.<br><br>
    <a href="https://cropwatch.unl.edu/common-rust-corn" target="_blank">Read More</a><br>
    <img src="https://cropwatch.unl.edu/images/common-rust-corn.jpg" width="300">
    """,

    'Corn_(maize)___Northern_Leaf_Blight': """
    Use resistant cultivars and apply fungicides at silking stage.<br><br>
    <a href="https://cropwatch.unl.edu/northern-corn-leaf-blight" target="_blank">Read More</a><br>
    <img src="https://cropwatch.unl.edu/images/northern-leaf-blight.jpg" width="300">
    """,

    'Corn_(maize)___healthy': """
    Corn is healthy. Maintain good crop rotation and monitor regularly.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/Maize_crop.jpg" width="300">
    """,

    'Grape___Black_rot': """
    Prune infected vines and apply fungicides such as mancozeb or myclobutanil.<br><br>
    <a href="https://grapes.extension.org/black-rot-of-grape/" target="_blank">Read More</a><br>
    <img src="https://grapes.extension.org/wp-content/uploads/2018/10/blackrot.jpg" width="300">
    """,

    'Grape___Esca_(Black_Measles)': """
    Remove infected wood, avoid pruning in wet conditions, and manage water stress.<br><br>
    <a href="https://plantpathology.ca.uky.edu/files/ppfs-fr-s-20.pdf" target="_blank">Read More</a><br>
    <img src="https://www.sciencedirect.com/science/article/pii/S0261219417302269/figures/fig1" width="300">
    """,

    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': """
    Improve air flow and apply fungicides like copper hydroxide.<br><br>
    <a href="https://www.extension.purdue.edu/extmedia/BP/BP-67-W.pdf" target="_blank">Read More</a><br>
    <img src="https://grapes.extension.org/wp-content/uploads/2018/10/leafspot.jpg" width="300">
    """,

    'Grape___healthy': """
    Grape vine is healthy. Maintain routine care and monitoring.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/bb/Grapes_in_Vineyard.jpg" width="300">
    """,

    'Orange___Haunglongbing_(Citrus_greening)': """
    Remove infected trees and control psyllid vectors. Use certified disease-free plants.<br><br>
    <a href="https://www.aphis.usda.gov/aphis/ourfocus/planthealth/plant-pest-and-disease-programs/pests-and-diseases/citrus/huanglongbing" target="_blank">Read More</a><br>
    <img src="https://www.ars.usda.gov/ARSUserFiles/00000000/hbl.jpg" width="300">
    """,

    'Peach___Bacterial_spot': """
    Apply copper sprays. Remove infected twigs and avoid overhead watering.<br><br>
    <a href="https://extension.umn.edu/plant-diseases/bacterial-spot-stone-fruit" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/bacterial-spot-peach.jpg" width="300">
    """,

    'Peach___healthy': """
    Tree is healthy. Maintain good sanitation and apply dormant sprays.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/fb/Peach_and_cross_section.jpg" width="300">
    """,

    'Pepper,_bell___Bacterial_spot': """
    Use copper-based bactericides. Avoid splashing water on leaves.<br><br>
    <a href="https://edis.ifas.ufl.edu/publication/PP121" target="_blank">Read More</a><br>
    <img src="https://www.vegetables.cornell.edu/files/2019/04/Bacterial-spot-on-pepper.jpg" width="300">
    """,

    'Pepper,_bell___healthy': """
    Plant is healthy. Ensure proper spacing and watering.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/1/13/Red_Bell_Pepper.jpg" width="300">
    """,

    'Potato___Early_blight': """
    Apply fungicides like chlorothalonil. Avoid overhead watering.<br><br>
    <a href="https://extension.umn.edu/diseases/early-blight-potato" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/early-blight-potato.jpg" width="300">
    """,

    'Potato___Late_blight': """
    Remove infected plants. Apply systemic fungicides like metalaxyl.<br><br>
    <a href="https://www.extension.purdue.edu/extmedia/BP/BP-83-W.pdf" target="_blank">Read More</a><br>
    <img src="https://www.syngenta-us.com/-/media/Images/Syngenta/Crop-Protection/Potato/PotatoLateBlight.jpg" width="300">
    """,

    'Potato___healthy': """
    Potato plant is healthy. Keep the soil well-drained and inspect regularly.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Potato_Plants.JPG" width="300">
    """,

    'Raspberry___healthy': """
    Raspberry is healthy. Maintain weed control and proper irrigation.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Raspberry_Bush.jpg" width="300">
    """,

    'Soybean___healthy': """
    Soybean is healthy. Regularly monitor for pests and ensure balanced nutrition.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/bf/Soybean_Plant.jpg" width="300">
    """,

    'Squash___Powdery_mildew': """
    Apply neem oil or sulfur fungicides. Improve air circulation.<br><br>
    <a href="https://extension.umn.edu/diseases/powdery-mildew" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/powdery-mildew-squash.jpg" width="300">
    """,
    'Strawberry___Leaf_scorch': """
    Remove infected leaves. Apply fungicides like myclobutanil. Ensure proper spacing for air flow.<br><br>
    <a href="https://extension.psu.edu/strawberry-leaf-scorch" target="_blank">Read More</a><br>
    <img src="https://extension.psu.edu/media/wysiwyg/leaf-scorch-strawberry.jpg" width="300">
    """,

    'Strawberry___healthy': """
    Strawberry plant is healthy. Use straw mulch and maintain proper watering.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/29/Strawberries_in_field.jpg" width="300">
    """,

    'Tomato___Bacterial_spot': """
    Use copper-based fungicides. Rotate crops and avoid wet foliage.<br><br>
    <a href="https://extension.umn.edu/vegetables/growing-tomatoes#bacterial-spot-278561" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/bacterial-spot-tomato.jpg" width="300">
    """,

    'Tomato___Early_blight': """
    Use fungicide sprays like chlorothalonil. Mulch around base to reduce spore splash.<br><br>
    <a href="https://extension.umn.edu/diseases/early-blight-tomato" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/early-blight-tomato.jpg" width="300">
    """,

    'Tomato___Late_blight': """
    Remove infected plants. Use resistant varieties and apply fungicides like chlorothalonil.<br><br>
    <a href="https://extension.umn.edu/diseases/late-blight" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/late-blight.jpg" width="300">
    """,

    'Tomato___Leaf_Mold': """
    Increase ventilation, avoid overhead watering, and apply fungicides like mancozeb.<br><br>
    <a href="https://extension.umn.edu/diseases/leaf-mold-tomato" target="_blank">Read More</a><br>
    <img src="https://www.gardeningknowhow.com/wp-content/uploads/2021/08/tomato-leaf-mold.jpg" width="300">
    """,

    'Tomato___Septoria_leaf_spot': """
    Remove infected leaves. Use fungicides such as chlorothalonil and maintain plant spacing.<br><br>
    <a href="https://ag.umass.edu/vegetable/fact-sheets/tomato-septoria-leaf-spot" target="_blank">Read More</a><br>
    <img src="https://ag.umass.edu/sites/ag.umass.edu/files/fact-sheets/images/septoria_leaf_spot.jpg" width="300">
    """,

    'Tomato___Spider_mites Two-spotted_spider_mite': """
    Spray with insecticidal soap or neem oil. Keep humidity high to deter mites.<br><br>
    <a href="https://extension.umn.edu/yard-and-garden-insects/spider-mites" target="_blank">Read More</a><br>
    <img src="https://extension.umn.edu/sites/extension.umn.edu/files/spider-mites-tomato.jpg" width="300">
    """,

    'Tomato___Target_Spot': """
    Remove affected leaves and apply broad-spectrum fungicides like chlorothalonil.<br><br>
    <a href="https://plantvillage.psu.edu/topics/tomato/infos/diseases_target-spot" target="_blank">Read More</a><br>
    <img src="https://plantvillage.psu.edu/static/images/target-spot-tomato.jpg" width="300">
    """,

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': """
    Control whiteflies with insecticides. Remove and destroy infected plants.<br><br>
    <a href="https://edis.ifas.ufl.edu/publication/VH056" target="_blank">Read More</a><br>
    <img src="https://www.researchgate.net/profile/Mohammad-Momeni/publication/353209260/figure/fig1/AS:1045351674261508@1626106569937/Symptoms-of-tomato-yellow-leaf-curl-virus-TYLCV-infection-on-tomato-plant.jpg" width="300">
    """,

    'Tomato___Tomato_mosaic_virus': """
    Remove infected plants and sterilize tools. Avoid smoking near plants.<br><br>
    <a href="https://ag.umass.edu/vegetable/fact-sheets/tomato-tobacco-tomato-mosaic-virus" target="_blank">Read More</a><br>
    <img src="https://www.greenlife.co.ke/wp-content/uploads/2020/08/tomato-mosaic-virus.jpg" width="300">
    """,

    'Tomato___healthy': """
    Tomato plant is healthy. Water at base and stake properly to avoid contact with soil.<br><br>
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e2/Tomato_plant.jpg" width="300">
    """
    }

      predicted_disease = class_name[result_index]
      treatment_advice = treatment_dict[predicted_disease]

      st.markdown(f"""
        <div style="background-color: #fff8e1; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
          <h4 style="color: #007acc;">🌱 Predicted Disease:</h4>
          <p style="font-size: 18px; font-weight: bold; color: black;">{predicted_disease}</p>
          <h4 style="color: #007acc;">🩺 Treatment Advice:</h4>
          <div style="font-size: 16px; color: black;">{treatment_advice}</div>
        </div>
        """, unsafe_allow_html=True)
