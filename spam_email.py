import joblib
import streamlit as st

# Load the pre-trained model and vectorizer
model = joblib.load('spam_email_detector.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_spam(text):
    # Transform the text using the loaded vectorizer
    text_transformed = vectorizer.transform([text])
    
    # Convert sparse matrix to dense array if necessary
    text_transformed = text_transformed.toarray()
    
    # Make prediction
    prediction = model.predict(text_transformed)
    
    return prediction[0]

def main():
    st.title('Spam Email Detector')
    
    user_input = st.text_area("Enter the email content here:")
    
    if st.button('Check for Spam'):
        result = predict_spam(user_input)
        
        if result == 1:
            st.write("The email is classified as SPAM.")
        else:
            st.write("The email is classified as HAM (legitimate).")
            
if __name__ == "__main__":
    main()