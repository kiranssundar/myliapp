import pandas as pd 
import numpy as np
import plotnine as p9
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st 
import plotly.graph_objects as go

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x==1, 1, 0)


ss = s[["web1h","income", "educ2", "par", "marital", "gender", "age"]]
ss = ss.rename(columns={"educ2":"education", "par":"parent", "marital":"married", "gender":"female"})
ss["sm_li"] = clean_sm(ss["web1h"])
ss["parent"] = clean_sm(ss["parent"])
ss["married"] = clean_sm(ss["married"])
ss["education"] = np.where(ss["education"] > 8, np.nan, ss["education"])
ss["income"] = np.where(ss["income"] > 9, np.nan, ss["income"])
ss["age"] = np.where(ss["age"] > 98, np.nan, ss["age"])
def clean_female(x):
    return np.where(x==2, 1, 0)
ss["female"] = clean_female(ss["female"])
ss = ss.drop(columns = ["web1h"])
ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X.values, 
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=51497)
# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

lr = LogisticRegression(class_weight = "balanced")

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


def li_pred_app(income, education, parent, married, female, age):
    
    person = [income, education, parent, married, female, age]
    # Predict class, given input features
    predicted_class = lr.predict([person])
    # Generate probability of positive class (=1)
    probs = lr.predict_proba([person])
    
    return predicted_class, probs 

#Streamlit user inputs
st.title("Kiran's LinkedIn User Prediction App")
st.markdown("Welcome to Kiran's LinkedIn User Prediction App!")
st.markdown("Want to find out if someone could be a LinkedIn user based on a few simple characteristics? Try making some selections below and press the Predict button when you're ready!")
income = st.slider("What is this person's income level? Hover over the help button to the right for more info.", 1, 9, help = "1 - Less than $10,000; 2 - $10,000-$19,999; 3 - $20,000-$29,999; 4 - $30,000-$39,999; 5 - $40,000-$49,999; 6 - $50,000-$74,999; 7 - $75,000-$99,999; 8 - $100,000-$149,999; 9 - $150,000+")
education = st.slider("What is the highest level of education that this person has completed? Hover over the help button to the right for more info.", 1, 8, help = "1 - Less than high school; 2 - High school incomplete; 3 - Graduated high school; 4 - Some college; 5 - 2-year associates degree; 6 - 4-year college/university degree; 7 - Some postgraduate/professional schooling; 8 - Postgraduate or professional degree (Master's, Doctorate, MD, etc.)")
parent = 1 if st.selectbox("Is this person a parent to a child under 18 who lives at home?", ["No","Yes"]) == "Yes" else 0 
married = 1 if st.selectbox("Is this person married?", ["No","Yes"]) == "Yes" else 0 
female = 1 if st.selectbox("Does this person identify as female or male?", ["Male","Female"]) == "Female" else 0 
age = st.slider("Age", 18, 97)

#User press predict button
if st.button("Predict", icon="🤔"):
    predicted_class, probs = li_pred_app(income, education, parent, married, female, age) 
    if predicted_class == 1:
            st.write("This person is predicted to be a LinkedIn user.")
            fig = go.Figure(go.Indicator(
                mode = "gauge",
                value = round(probs[0][1]*100,2),
                title = {'text': "What are the chances that this person is a LinkedIn user?"},
                gauge = {"axis": {"range": [0, 100]},
                "steps": [
                {"range": [0, 33], "color":"red"},
                {"range": [34, 66], "color":"gray"},
                {"range": [67, 100], "color":"lightgreen"}
            ],
            "bar":{"color":"yellow"}}
             ))
            fig
            st.write(f"There is a {round(probs[0][1] * 100, 2)}% chance that this person is a LinkedIn user.") 
    else: 
            st.write("This person is not predicted to be a LinkedIn user.")
            fig = go.Figure(go.Indicator(
                mode = "gauge",
                value = round(probs[0][1]*100,2),
                title = {'text': "What is the probability that this person is a LinkedIn user?"},
                gauge = {"axis": {"range": [0, 100]},
                "steps": [
                {"range": [0, 33], "color":"red"},
                {"range": [34, 66], "color":"gray"},
                {"range": [67, 100], "color":"lightgreen"}
            ],
            "bar":{"color":"yellow"}}
             ))
            fig 
            st.write(f"There is a {round(probs[0][1] * 100, 2)}% chance that this person is a LinkedIn user.") 
