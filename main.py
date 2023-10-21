from flask import Flask,render_template,request
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models  import load_model
model=load_model("model/v3_red_cott_dis.h5")

print("Model Loded")


def pred_cot_disease(cott_plant):
    test_image=load_img(cott_plant,target_size=(150,150))
    print("get image for prediction")
    test_image=img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # get the index of max value

    if pred == 0:
        return "Healthy Cotton Plant", 'healthy_plant_leaf.html'  # if index 0 burned leaf
    elif pred == 1:
        return 'Diseased Cotton Plant', 'disease_plant.html'  # # if index 1
    elif pred == 2:
        return 'Healthy Cotton Plant', 'healthy_plant.html'  # if index 2  fresh leaf
    else:
        return "Healthy Cotton Plant", 'healthy_plant.html'  # if index 3

    # ------------>>pred_cot_dieas<<--end

app=Flask(__name__)

#render index.html page
@app.route("/",method=['GET','POST'])
def home():
    return render_template('index.html')


# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_disease(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)



#     # Define CSS styles for the table
#     table_styles = """
#     <style>
#         th {
#             background-color: black; /* Background color for table headers */
#             color: white; /* Text color for table headers */
#         }
#         td {
#             background-color: black; /* Background color for table cells */
#             color: white; /* Text color for table cells */
#         }
#     </style>
#     """
#
#     # Display the CSS-styled table with formatted ratings and a black background
#     st.write(table_styles, unsafe_allow_html=True)
#     st.table(top_s[['', '']].style.format({'Rating': '{:.1f}'}))

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False)
