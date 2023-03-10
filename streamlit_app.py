import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the models (assuming they have been trained and saved)
model_bert = tf.keras.models.load_model('path_to_bert_model.h5')
model_efficientnet = tf.keras.models.load_model('path_to_efficientnet_model.h5')

# Get the class names (assuming they have been saved along with the models)
with open('path_to_class_names.txt', 'r') as f:
    class_names = f.read().splitlines()

def predict_class(image, model):
    # Preprocess the input image
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image){"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import pandas as pd\nimport numpy as np\nfrom tqdm.auto import tqdm\nimport tensorflow as tf\nfrom transformers import BertTokenizer\nimport os\nimport cv2\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\nfrom random import shuffle\nimport tensorflow.keras.backend as K\nfrom tensorflow.keras.utils import to_categorical\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.model_selection import train_test_split\nimport matplotlib.pyplot as plt\nimport tensorflow_hub as hub\nfrom tensorflow.keras import layers\nimport datetime\nfrom array import *\nfrom tensorflow.keras.models import Model\nfrom transformers import TFBertModel, BertConfig\nfrom PIL import Image\nimport pytesseract\nimport re\nimport nltk\nfrom nltk.stem import WordNetLemmatizer \nfrom nltk.corpus import stopwords \nfrom nltk.tokenize import word_tokenize\nimport zipfile\nimport tempfile\nfrom PIL import Image #, ImageDraw, ImageFont\n#from PIL import ImageOps\nnltk.download('wordnet')\nnltk.download('stopwords')\nnltk.download('punkt')\nnltk.download('omw-1.4')\nstop_words = set(stopwords.words('english')) \n\n\n\ndf = pd.read_csv('your_file.csv')\nIMAGE_SIZE = 1800\nBINARY_THREHOLD = 180\n\ndef process_image_for_ocr(file_path):\n    im_new = set_image_dpi(file_path)\n#     im_new = remove_noise_and_smooth(temp_filename)\n#     inv_img=invert_image(temp_filename)\n#     crop_image=crop_img(im_new, 0.8)\n#     rot_img = rotate(im_new)\n    return im_new\n\ndef invert_image(file):\n    img = cv2.imread(file,0)\n    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]\n    img = cv2.bitwise_not(img)\n    return img\n\ndef set_image_dpi(file_path):\n    im = Image.open(file_path)\n    length_x, width_y = im.size\n    factor = max(1, int(IMAGE_SIZE / length_x))\n    size = factor * length_x, factor * width_y\n    # size = (1800, 1800)\n    im_resized = im.resize(size, Image.Resampling.LANCZOS)\n    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')\n    temp_filename = temp_file.name\n    im_resized.save(temp_filename, dpi=(200, 200))\n    return im_resized\n\ndef image_smoothening(img):\n    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)\n    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n    blur = cv2.GaussianBlur(th2, (1, 1), 0)\n    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n    return th3\n\ndef remove_noise_and_smooth(file_name):\n    img = cv2.imread(file_name, 0)\n    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,\n                                     3)\n    kernel = np.ones((1, 1), np.uint8)\n    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)\n    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)\n    img = image_smoothening(img)\n    or_image = cv2.bitwise_or(img, closing)\n    img = cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)\n    im_pil = Image.fromarray(img)\n    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')\n    temp_filename = temp_file.name\n    im_pil.save(temp_filename, dpi=(200, 200))\n    return img\n\ndef rotate(im,scale = 1.0):\n    try:\n        newdata=pytesseract.image_to_osd(im, config='--psm 0 -c min_characters_to_try=5')\n        ra=re.search('(?<=Rotate: )\\d+', newdata).group(0)\n    except:\n        ra='0'\n    if(ra!='0'):\n        if(int(ra)==90):\n            angle=270\n        if(int(ra)==270):\n            angle=-90        \n        else:\n            angle=360-int(ra)\n        height, width = im.shape[:2] # image shape has 3 dimensions\n        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape\n\n        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)\n\n        # rotation calculates the cos and sin, taking absolutes of those.\n        abs_cos = abs(rotation_mat[0,0]) \n        abs_sin = abs(rotation_mat[0,1])\n\n        # find the new width and height bounds\n        bound_w = int(height * abs_sin + width * abs_cos)\n        bound_h = int(height * abs_cos + width * abs_sin)\n\n        # subtract old image center (bringing image back to origo) and adding the new image center coordinates\n        rotation_mat[0, 2] += bound_w/2 - image_center[0]\n        rotation_mat[1, 2] += bound_h/2 - image_center[1]\n\n        # rotate image with the new bounds and translated rotation matrix\n        rotated_mat = cv2.warpAffine(im, rotation_mat, (bound_w, bound_h))\n        return rotated_mat\n    else:\n        return im\n\ndef crop_img(img, scale=1.0):\n    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2\n    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale\n    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2\n    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2\n    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]\n    return img_cropped\n\ndef word_extractor(img):\n    words = pytesseract.image_to_string(img)\n#     ocr_df = pytesseract.image_to_data(img, output_type='data.frame')\n#     ocr_df = ocr_df.dropna().reset_index(drop=True)\n#     float_cols = ocr_df.select_dtypes('float').columns\n#     ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)\n#     ocr_df = ocr_df.replace(r'^\\s*$', np.nan, regex=True)\n#     words = ' '.join([str(word) for word in ocr_df.text if str(word) != 'nan'])\n    return words\n\ndef normalise_text(text):\n    text = text.lower() # lowercase\n    text=re.sub('[^A-Za-z0-9]+', ' ', text)\n    return text\n\ndef tok_lem(text):\n    word_tokens = word_tokenize(text)\n    filtered_sentence = [w for w in word_tokens if not w in stop_words] \n    lemmatizer = WordNetLemmatizer()\n    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_sentence])\n    return lemmatized_output\n\n\ndf['text']=''\ni=0\nfor path in tqdm(df.address):\n    img=process_image_for_ocr(path)\n    extracted=word_extractor(path)\n    norm=normalise_text(extracted)\n    text = tok_lem(norm)\n    df.at[i,'text']=norm\n    i=i+1\n\n\n","metadata":{"_uuid":"fb75e65e-bbb7-4be4-b4f0-d1513f9622cc","_cell_guid":"fc68a898-03a6-4c11-bd77-7a151e17b5bd","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}
    predictions = np.squeeze(predictions)
    class_index = np.argmax(predictions)
    class_probability = predictions[class_index]

    return class_names[class_index], class_probability

# Set up the Streamlit app
st.set_page_config(page_title="Image Classifier", page_icon=":camera:", layout="wide")

# Add a title
st.title("Image Classifier")

# Add an image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    image = Image.open(uploaded_image)

    # Show the image
    st.image(image, use_column_width=True)

    # Predict the class using BERT
    class_bert, prob_bert = predict_class(image, model_bert)
    st.write(f"Prediction using BERT: {class_bert} ({prob_bert * 100:.2f}%)")

    # Predict the class using EfficientNet
    class_efficientnet, prob_efficientnet = predict_class(image, model_efficientnet)
    st.write(f"Prediction using EfficientNet: {class_efficientnet} ({prob_efficientnet * 100:.2f}%)")
