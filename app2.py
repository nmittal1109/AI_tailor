import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
# Function to parse the user input and return a dictionary

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

#@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" style='margin-top:-100px;margin-left:250px;height:150px' />
            
        </a>'''
    return html_code

gif_html = get_img_with_href('ai-tailor_logo.jpg', 'https://docs.streamlit.io')

df = pd.read_csv('new_export1.csv')

df1=df[df['special note']!='woman']

df1['hips (order sheet)']=pd.to_numeric(df1['hips (order sheet)'],errors='coerce')
df1['neck (order sheet)']=pd.to_numeric(df1['neck (order sheet)'],errors='coerce')
df=df1.copy()
def parse_input(user_input):
    lines = user_input.strip().split('\n')
    data = {line.split(' : ')[0]: line.split(' : ')[1] for line in lines}
    return data


def predict_neck_order(test_row):
    # Creating the condition based on test row
    condition = (np.abs(df['neck'] - test_row['neck']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Checking if there are more than 5 profiles with the same fit
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'neck (order sheet)'
    else:
        return filtered_df['neck (order sheet)'].mean()

# Creating a new column 'predicted_neck_order' based on the function



def predict_chest_order(test_row):
    # Creating the initial condition based on chest, height, and weight
    condition = (np.abs(df['chest'] - test_row['chest']) <= 1) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Apply Fit filter
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # Apply Stomach filter
    if filtered_df[np.abs(filtered_df['stomach'] - test_row['stomach']) <= 3].shape[0] >= 5:
        filtered_df = filtered_df[np.abs(filtered_df['stomach'] - test_row['stomach']) <= 3]

    # Apply Country filter
    if test_row['country'] == 'US':
        if filtered_df[filtered_df['country'] == 'US'].shape[0] >= 5:
            filtered_df = filtered_df[filtered_df['country'] == 'US']
    elif test_row['country'] == 'AU':
        if filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')].shape[0] >= 5:
            filtered_df = filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'chest (order sheet)'
    else:
        return filtered_df['chest (order sheet)'].mean()



def predict_stomach_order(test_row):
    # Creating the initial condition based on chest, height, and weight
    condition = (np.abs(df['stomach'] - test_row['stomach']) <= 1) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Apply Fit filter
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # Apply Stomach filter
    if filtered_df[np.abs(filtered_df['chest'] - test_row['chest']) <= 3].shape[0] >= 5:
        filtered_df = filtered_df[np.abs(filtered_df['chest'] - test_row['chest']) <= 3]

    # Apply Country filter
    if test_row['country'] == 'US':
        if filtered_df[filtered_df['country'] == 'US'].shape[0] >= 5:
            filtered_df = filtered_df[filtered_df['country'] == 'US']
    elif test_row['country'] == 'AU':
        if filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')].shape[0] >= 5:
            filtered_df = filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'chest (order sheet)'
    else:
        return filtered_df['stomach (order sheet)'].mean()

    
def predict_hips_order(test_row):
    # Creating the initial condition based on chest, height, and weight
    condition = (np.abs(df['hips'] - test_row['hips']) <= 1) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Apply Fit filter
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # Apply Stomach filter
    if filtered_df[np.abs(filtered_df['stomach'] - test_row['stomach']) <= 3].shape[0] >= 5:
        filtered_df = filtered_df[np.abs(filtered_df['chest'] - test_row['chest']) <= 3]

    # Apply Country filter
    if test_row['country'] == 'US':
        if filtered_df[filtered_df['country'] == 'US'].shape[0] >= 5:
            filtered_df = filtered_df[filtered_df['country'] == 'US']
    elif test_row['country'] == 'AU':
        if filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')].shape[0] >= 5:
            filtered_df = filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'chest (order sheet)'
    else:
        return filtered_df['hips (order sheet)'].mean()
    
    
def predict_shirt_length_order(test_row):
    # Creating the condition based on test row
    condition = (np.abs(df['shirt-length'] - test_row['shirt-length']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Checking if there are more than 5 profiles with the same fit
    if filtered_df[filtered_df['shirt style'] == test_row['shirt style']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['shirt style'] == test_row['shirt style']]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'neck (order sheet)'
    else:
        return filtered_df['shirt-length (order sheet)'].mean()
    
    
def predict_sleeve_length_order(test_row):
    # Creating the condition based on test row
    condition = (np.abs(df['arm-length'] - test_row['arm-length']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Checking if there are more than 5 profiles with the same fit
    #if filtered_df[filtered_df['shirt style'] == test_row['shirt style']].shape[0] >= 5:
     #   filtered_df = filtered_df[filtered_df['shirt style'] == test_row['shirt style']]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'neck (order sheet)'
    else:
        return filtered_df['arm-length (order sheet)'].mean()
    
    
    
def predict_shoulders_order(test_row):
    # Creating the condition based on test row
    condition = (np.abs(df['shoulders'] - test_row['shoulders']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Checking if there are more than 5 profiles with the same fit
    if filtered_df[(np.abs(filtered_df['chest'] - test_row['chest']) <= 2)].shape[0] >= 5:
         filtered_df = filtered_df[(np.abs(filtered_df['chest'] - test_row['chest']) <= 2)]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'neck (order sheet)'
    else:
        return filtered_df['shoulders (order sheet)'].mean()
    
    
    
    
def predict_wrist_order(test_row):
    # Creating the initial condition based on chest, height, and weight
    condition = (np.abs(df['wrist / cuff'] - test_row['wrist / cuff']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Apply Fit filter
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # Apply Stomach filter
    #if filtered_df[np.abs(filtered_df['chest'] - test_row['chest']) <= 3].shape[0] >= 5:
     #   filtered_df = filtered_df[np.abs(filtered_df['chest'] - test_row['chest']) <= 3]

    # Apply Country filter
    if test_row['country'] == 'US':
        if filtered_df[filtered_df['country'] == 'US'].shape[0] >= 5:
            filtered_df = filtered_df[filtered_df['country'] == 'US']
    elif test_row['country'] == 'AU':
        if filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')].shape[0] >= 5:
            filtered_df = filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'chest (order sheet)'
    else:
        return filtered_df['wrist (order sheet)'].mean()
    
    
    
    
def predict_bicep_order(test_row):
    # Creating the initial condition based on wrist/cuff, height, and weight
    condition = (np.abs(df['bicep'] - test_row['bicep']) <= 0.5) &\
                (np.abs(df['height (cm)'] - test_row['height (cm)'])/test_row['height (cm)'] <= 0.0214) &\
                (np.abs(df['weight (kg)'] - test_row['weight (kg)'])/test_row['weight (kg)'] <= 0.082)

    filtered_df = df[condition]

    # Apply Fit filter
    if filtered_df[filtered_df['fit'] == test_row['fit']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['fit'] == test_row['fit']]

    # Apply Build filter
    if filtered_df[filtered_df['build'] == test_row['build']].shape[0] >= 5:
        filtered_df = filtered_df[filtered_df['build'] == test_row['build']]

    # Apply Country filter
    if test_row['country'] == 'US':
        if filtered_df[filtered_df['country'] == 'US'].shape[0] >= 5:
            filtered_df = filtered_df[filtered_df['country'] == 'US']
    elif test_row['country'] == 'AU':
        if filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')].shape[0] >= 5:
            filtered_df = filtered_df[(filtered_df['country'] != 'US') & (filtered_df['country'] != 'SG')]

    # If the number of profiles is less than 2, return 'ERROR'
    if filtered_df.shape[0] < 2:
        return 'ERROR'

    # Else return the average of 'wrist (order sheet)'
    else:
        return filtered_df['bicep (order sheet)'].mean()

st.title('')
st.markdown(gif_html, unsafe_allow_html=True)

option = st.selectbox('Choose Input Method', ['Upload CSV', 'Input Manually'])

if option == 'Upload CSV':
    file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if file is not None:
        df_test = pd.read_csv(file)
        df_test['predicted_neck_order'] = df_test.apply(predict_neck_order, axis=1)
        df_test['predicted_chest_order'] = df_test.apply(predict_chest_order, axis=1)
        df_test['predicted_neck_order'] = df_test.apply(predict_neck_order, axis=1)
        df_test['predicted_chest_order'] = df_test.apply(predict_chest_order, axis=1)
        df_test['predicted_stomach_order'] = df_test.apply(predict_stomach_order, axis=1)
        df_test['predicted_hips_order'] = df_test.apply(predict_hips_order, axis=1)
        df_test['predicted_shirt_length_order'] = df_test.apply(predict_shirt_length_order, axis=1)
        df_test['predicted_sleeve_length_order'] = df_test.apply(predict_sleeve_length_order, axis=1)
        df_test['predicted_shoulder_order'] = df_test.apply(predict_shoulders_order, axis=1)
        df_test['predicted_wrist_order'] = df_test.apply(predict_wrist_order, axis=1)
        df_test['predicted_bicep_order'] = df_test.apply(predict_bicep_order, axis=1)
        st.dataframe(df_test)
        df_test.to_csv('predictions.csv')
        st.download_button(label="Download Predictions as CSV", data=df_test.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

    
# Text area for user input
else:
    user_input = st.text_area('Enter your data:', height=300)

# Button to process the input
    process_button = st.button('Predict')

    if process_button:
        if user_input:
            # Parse the user input
            data_dict = parse_input(user_input)
            

            # Convert the dictionary to a DataFrame and display it
            df_test = pd.DataFrame([data_dict])
            #st.write(df_test.dtypes)
            df_test.columns=['country', 'Cuff', 'Measurement Type', 'Profile-name', 'Value-type',
                        'neck', 'chest', 'stomach', 'hips', 'shirt-length', 'arm-length',
                        'shoulders', 'bicep', 'wrist / cuff', 'height (cm)', 'weight (kg)', 'build', 'fit',
                        'shirt style', 'Shoulder-type']
            
            df_test['neck']=df_test['neck'].astype('float')
            df_test['chest']=df_test['chest'].astype('float')
            df_test['stomach']=df_test['stomach'].astype('float')
            df_test['hips']=df_test['hips'].astype('float')
            df_test['shirt-length']=df_test['shirt-length'].astype('float')
            df_test['arm-length']=df_test['arm-length'].astype('float')
            df_test['shoulders']=df_test['shoulders'].astype('float')
            df_test['bicep']=df_test['bicep'].astype('float')
            df_test['wrist / cuff']=df_test['wrist / cuff'].astype('float')
            
            df_test['height (cm)'] = df_test['height (cm)'].str.split(' cm').str[0].astype('float')
            df_test['weight (kg)'] = df_test['weight (kg)'].str.split(' kg').str[0].astype('float')
            df_test['predicted_neck_order'] = df_test.apply(predict_neck_order, axis=1)
            df_test['predicted_chest_order'] = df_test.apply(predict_chest_order, axis=1)
            df_test['predicted_neck_order'] = df_test.apply(predict_neck_order, axis=1)
            df_test['predicted_chest_order'] = df_test.apply(predict_chest_order, axis=1)
            df_test['predicted_stomach_order'] = df_test.apply(predict_stomach_order, axis=1)
            df_test['predicted_hips_order'] = df_test.apply(predict_hips_order, axis=1)
            df_test['predicted_shirt_length_order'] = df_test.apply(predict_shirt_length_order, axis=1)
            df_test['predicted_sleeve_length_order'] = df_test.apply(predict_sleeve_length_order, axis=1)
            df_test['predicted_shoulder_order'] = df_test.apply(predict_shoulders_order, axis=1)
            df_test['predicted_wrist_order'] = df_test.apply(predict_wrist_order, axis=1)
            df_test['predicted_bicep_order'] = df_test.apply(predict_bicep_order, axis=1)
            st.dataframe(df_test)
            df_test1=df_test
            df_test1 = df_test1.filter(like='predicted')
            df_test1 = df_test1.round(2)
            data_dict = df_test1.iloc[0].to_dict()

            # Initialize an empty string
            text = ''

            # Iterate over the dictionary to format it as a string
            for key, value in data_dict.items():
                text += f'{key}: {value}\n'  # '\n' is for a new line

            # Display the text in Streamlit
            st.text(text)
            df_test.to_csv('predictions.csv')
            st.download_button(label="Download Predictions as CSV", data=df_test.to_csv(index=False), file_name='predictions.csv', mime='text/csv')


            #st.dataframe(df_test)
        else:
            st.write('Please input some data.')

