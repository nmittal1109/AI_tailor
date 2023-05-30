import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('new_export1.csv')


df1=df[df['special note']!='woman']
df1['hips (order sheet)']=pd.to_numeric(df1['hips (order sheet)'],errors='coerce')
df1['neck (order sheet)']=pd.to_numeric(df1['neck (order sheet)'],errors='coerce')

def predict_collar(df, neck, height,weight, fit, chest, stomach, hips):
    filter_condition = (
        (df['fit'] == fit) & 
        (df['neck'].between(neck-1, neck+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) & 
        (df['chest'].between(chest-1, chest+1)) & 
        (df['stomach'].between(stomach-1, stomach+1)) & 
        (df['hips'].between(hips-1, hips+1))
    )
    return apply_prediction_logic(df, filter_condition, neck, 'neck (order sheet)', 'neck')

def predict_chest(df, chest, height,weight, fit, stomach, country, build):
    filter_condition = (
        (df['fit'] == fit) & 
        (df['country'] == country) & 
        (df['build'] == build) & 
        (df['chest'].between(chest-1, chest+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) &  
        (df['stomach'].between(stomach-1, stomach+1))
    )
    return apply_prediction_logic(df, filter_condition, chest, 'chest (order sheet)', 'chest')

def predict_stomach(df, stomach, height,weight, fit, chest, hips, country):
    filter_condition = (
        (df['fit'] == fit) & 
        (df['country'] == country) & 
        (df['stomach'].between(stomach-1, stomach+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) &  
        (df['chest'].between(chest-1, chest+1)) & 
        (df['hips'].between(hips-1, hips+1))
    )
    return apply_prediction_logic(df, filter_condition, stomach, 'stomach (order sheet)', 'stomach')

def predict_hips(df, hips, height,weight, fit, chest, stomach, shirt_style, country):
    filter_condition = (
        (df['fit'] == fit) & 
        (df['shirt style'] == shirt_style) & 
        (df['country'] == country) & 
        (df['hips'].between(hips-1, hips+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) & 
        (df['chest'].between(chest-1, chest+1)) & 
        (df['stomach'].between(stomach-1, stomach+1))
    )
    return apply_prediction_logic(df, filter_condition, hips, 'hips (order sheet)', 'hips')

def predict_sleeve_length(df, arm_length, height, weight, chest, country):
    filter_condition = (
        (df['arm-length'].between(arm_length-1, arm_length+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) & 
        (df['chest'].between(chest-1, chest+1)) & 
        (df['country'] == country)
    )
    return apply_prediction_logic(df, filter_condition, arm_length, 'arm-length (order sheet)', 'arm-length')

def predict_shirt_length(df, shirt_length, height, shirt_style, weight):
    filter_condition = (
        (df['shirt-length'].between(shirt_length-1, shirt_length+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['shirt style'] == shirt_style) & 
        (df['weight (kg)'].between(weight-1, weight+1))
    )
    return apply_prediction_logic(df, filter_condition, shirt_length, 'shirt-length (order sheet)', 'shirt-length')

def predict_shoulders(df, shoulders, weight, chest, height, country, build):
    filter_condition = (
        (df['shoulders'].between(shoulders-1, shoulders+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) & 
        (df['chest'].between(chest-1, chest+1)) & 
         
        (df['country'] == country) & 
        (df['build'] == build)
    )
    return apply_prediction_logic(df, filter_condition, shoulders, 'shoulders (order sheet)', 'shoulders')

def predict_armhole(df, height,weight, chest, bicep):
    filter_condition = (
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1))  & 
        (df['chest'].between(chest-1, chest+1)) & 
        (df['bicep'].between(bicep-1, bicep+1))
    )
    return apply_prediction_logic(df, filter_condition, bicep, 'armhole (order sheet)', 'bicep')

def predict_bicep(df, bicep, height,weight, country, build):
    filter_condition = (
        (df['bicep'].between(bicep-1, bicep+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) &  
        (df['country'] == country) & 
        (df['build'] == build)
    )
    return apply_prediction_logic(df, filter_condition, bicep, 'bicep (order sheet)', 'bicep')

def predict_cuff(df, wrist_cuff, height,weight, bicep):
    filter_condition = (
        (df['wrist / cuff'].between(wrist_cuff-1, wrist_cuff+1)) & 
        (df['height (cm)'].between(height-1, height+1)) & 
        (df['weight (kg)'].between(weight-1, weight+1)) &  
        (df['bicep'].between(bicep-1, bicep+1))
    )
    return apply_prediction_logic(df, filter_condition, wrist_cuff, 'wrist (order sheet)', 'wrist / cuff')




def apply_prediction_logic(df, filter_condition, measurement, order_sheet_col, measurement_col):
    filtered_data = df[filter_condition]
    if filtered_data.shape[0] <1:
        predicted_measurement = 'error'
    else:
        average_difference = (filtered_data[order_sheet_col] - filtered_data[measurement_col]).mean()
        predicted_measurement = measurement + average_difference
    return predicted_measurement

st.title('Measurement Predictor')

option = st.selectbox('Choose Input Method', ['Upload CSV', 'Input Manually'])

if option == 'Upload CSV':
    file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if file is not None:
        df_test = pd.read_csv(file)
        df_test['predicted_chest'] = df_test.apply(lambda row: predict_chest(df1, row['chest'], row['height (cm)'], row['weight (kg)'], row['fit'], row['stomach'], row['country'], row['build']), axis=1)
#neck, height,weight, fit, chest, stomach, hips
        df_test['predicted_collar'] = df_test.apply(lambda row: predict_collar(df1, row['neck'], row['height (cm)'], row['weight (kg)'], row['fit'],row['chest'], row['stomach'], row['hips']), axis=1)
        #df, stomach, height,weight, fit, chest, hips, country
        df_test['predicted_stomach'] = df_test.apply(lambda row: predict_stomach(df1, row['stomach'], row['height (cm)'], row['weight (kg)'], row['fit'], row['chest'], row['hips'],row['country']), axis=1)

        #df, arm_length, height, weight, chest, country
        df_test['predicted_sleeve_length'] = df_test.apply(lambda row: predict_sleeve_length(df1, row['arm-length'], row['height (cm)'], row['weight (kg)'], row['chest'], row['country']), axis=1)
        #df, shirt_length, height, shirt_style, weight)
        df_test['predicted_shirt_length'] = df_test.apply(lambda row: predict_shirt_length(df1, row['shirt-length'], row['height (cm)'], row['shirt style'], row['weight (kg)']), axis=1)
        #df, shoulders, weight, chest, height, country, build
        df_test['predicted_shoulders'] = df_test.apply(lambda row: predict_shoulders(df1, row['shoulders'], row['weight (kg)'], row['chest'], row['height (cm)'], row['country'], row['build']), axis=1)
        #df, height,weight, chest, bicep
        df_test['predicted_armhole'] = df_test.apply(lambda row: predict_armhole(df1, row['height (cm)'],row['weight (kg)'], row['chest'], row['bicep']), axis=1)
        #df, bicep, height,weight, country, build
        df_test['predicted_bicep'] = df_test.apply(lambda row: predict_bicep(df1, row['bicep'], row['height (cm)'],row['weight (kg)'], row['country'], row['build']), axis=1)
        #df, wrist_cuff, height,weight, bicep
        df_test['predicted_cuff'] = df_test.apply(lambda row: predict_cuff(df1, row['wrist / cuff'], row['height (cm)'],row['weight (kg)'], row['bicep']), axis=1)

        st.write(df_test)
        df_test.to_csv('predictions.csv')
        st.download_button(label="Download Predictions as CSV", data=df_test.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

    


else:
# create input boxes for manual input
    chest = st.number_input('Input Chest Measurement')
    neck = st.number_input('Input Neck Measurement')
    hips = st.number_input('Input Hips Measurement')
    height = st.number_input('Input Height Measurement')
    shirt_style = st.text_input('Input Shirt Style')
    bicep = st.number_input('Input Bicep Measurement')
    wirst_cuff = st.number_input('Input Wrist / Cuff Measurement')
    shirt_length=   st.number_input('Input Shirt Length Measurement')
    arm_length = st.number_input('Input Arm Length Measurement')
    shoulders = st.number_input('Input Shoulders Measurement')
    weight = st.number_input('Input Weight Measurement')
    fit = st.text_input('Input Fit')
    stomach = st.number_input('Input Stomach Measurement')
    country = st.text_input('Input Country')
    build = st.text_input('Input Build')

        # construct a dictionary and convert it into a dataframe
    data = {'chest': [chest], 'height (cm)': [height], 'weight (kg)': [weight], 'fit': [fit], 'stomach': [stomach], 'country': [country], 'build': [build]}
    df_test = pd.DataFrame(data)
        
    df_test['predicted_chest'] = df_test.apply(lambda row: predict_chest(df1, row['chest'], row['height (cm)'], row['weight (kg)'], row['fit'], row['stomach'], row['country'], row['build']), axis=1)
#neck, height,weight, fit, chest, stomach, hips
    df_test['predicted_collar'] = df_test.apply(lambda row: predict_collar(df1, row['neck'], row['height (cm)'], row['weight (kg)'], row['fit'],row['chest'], row['stomach'], row['hips']), axis=1)
        #df, stomach, height,weight, fit, chest, hips, country
    df_test['predicted_stomach'] = df_test.apply(lambda row: predict_stomach(df1, row['stomach'], row['height (cm)'], row['weight (kg)'], row['fit'], row['chest'], row['hips'],row['country']), axis=1)

        #df, arm_length, height, weight, chest, country
    df_test['predicted_sleeve_length'] = df_test.apply(lambda row: predict_sleeve_length(df1, row['arm-length'], row['height (cm)'], row['weight (kg)'], row['chest'], row['country']), axis=1)
        #df, shirt_length, height, shirt_style, weight)
    df_test['predicted_shirt_length'] = df_test.apply(lambda row: predict_shirt_length(df1, row['shirt-length'], row['height (cm)'], row['shirt style'], row['weight (kg)']), axis=1)
        #df, shoulders, weight, chest, height, country, build
    df_test['predicted_shoulders'] = df_test.apply(lambda row: predict_shoulders(df1, row['shoulders'], row['weight (kg)'], row['chest'], row['height (cm)'], row['country'], row['build']), axis=1)
        #df, height,weight, chest, bicep
    df_test['predicted_armhole'] = df_test.apply(lambda row: predict_armhole(df1, row['height (cm)'],row['weight (kg)'], row['chest'], row['bicep']), axis=1)
        #df, bicep, height,weight, country, build
    df_test['predicted_bicep'] = df_test.apply(lambda row: predict_bicep(df1, row['bicep'], row['height (cm)'],row['weight (kg)'], row['country'], row['build']), axis=1)
        #df, wrist_cuff, height,weight, bicep
    df_test['predicted_cuff'] = df_test.apply(lambda row: predict_cuff(df1, row['wrist / cuff'], row['height (cm)'],row['weight (kg)'], row['bicep']), axis=1)
  # assuming make_predictions is a function that applies all your prediction functions to the dataframe
    st.write(df_test)
    df_test.to_csv('predictions.csv')
    st.download_button(label="Download Predictions as CSV", data=df_test.to_csv(index=False), file_name='predictions.csv', mime='text/csv')
 
