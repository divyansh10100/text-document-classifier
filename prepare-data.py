import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from array import *
from PIL import Image
import pytesseract
import re
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import zipfile
import tempfile
from PIL import Image #, ImageDraw, ImageFont
#from PIL import ImageOps
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english')) 



df = pd.read_csv('your_file.csv')
IMAGE_SIZE = 1800
BINARY_THREHOLD = 180

def process_image_for_ocr(file_path):
    im_new = set_image_dpi(file_path)
#     im_new = remove_noise_and_smooth(temp_filename)
#     inv_img=invert_image(temp_filename)
#     crop_image=crop_img(im_new, 0.8)
#     rot_img = rotate(im_new)
    return im_new

def invert_image(file):
    img = cv2.imread(file,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.bitwise_not(img)
    return img

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.Resampling.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(200, 200))
    return im_resized

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    img = cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_pil.save(temp_filename, dpi=(200, 200))
    return img

def rotate(im,scale = 1.0):
    try:
        newdata=pytesseract.image_to_osd(im, config='--psm 0 -c min_characters_to_try=5')
        ra=re.search('(?<=Rotate: )\d+', newdata).group(0)
    except:
        ra='0'
    if(ra!='0'):
        if(int(ra)==90):
            angle=270
        if(int(ra)==270):
            angle=-90        
        else:
            angle=360-int(ra)
        height, width = im.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(im, rotation_mat, (bound_w, bound_h))
        return rotated_mat
    else:
        return im

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def word_extractor(img):
    words = pytesseract.image_to_string(img)
#     ocr_df = pytesseract.image_to_data(img, output_type='data.frame')
#     ocr_df = ocr_df.dropna().reset_index(drop=True)
#     float_cols = ocr_df.select_dtypes('float').columns
#     ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
#     ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
#     words = ' '.join([str(word) for word in ocr_df.text if str(word) != 'nan'])
    return words

def normalise_text(text):
    text = text.lower() # lowercase
    text=re.sub('[^A-Za-z0-9]+', ' ', text)
    return text

def tok_lem(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_sentence])
    return lemmatized_output


df['text']=''
i=0
for path in tqdm(df.address):
    img=process_image_for_ocr(path)
    extracted=word_extractor(path)
    norm=normalise_text(extracted)
    text = tok_lem(norm)
    df.at[i,'text']=norm
    i=i+1
