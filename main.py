import streamlit as st
from transformers import pipeline
import tf
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
)


def sa(text):
    out = distilled_student_sentiment_classifier(text)[0]
    ans = 'positive'
    x = 0
    for dic in out:
        if dic['score'] > x:
            x = dic['score']
            ans = dic['label']
    return ans


def main():
    st.title("Borhan")
    html_temp = """
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Sentiment Analysis</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    text = st.text_input('Enter The Text')

    pos_html = """  
          <div style="background-color:#F4D03F;padding:10px >
           <h2 style="color:white;text-align:center;"> Positive!</h2>
           </div>
        """
    neg_html = """  
          <div style="background-color:#F08080;padding:10px >
           <h2 style="color:black ;text-align:center;"> Negative!</h2>
           </div>
        """
    if st.button('Predict'):
        if sa(text) == 'positive':
            st.markdown(pos_html, unsafe_allow_html=True)
        else:
            st.markdown(neg_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()