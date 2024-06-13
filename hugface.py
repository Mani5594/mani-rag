from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from PIL import Image
import streamlit as st
import requests
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# img2text
def image2text(file):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text=image_to_text(file)[0]["generated_text"]
    return text


# llm
def generate_story(scenario):
    template = """
    you are a storu teller;
    you can generate a short story based on a simple narrative, the story should be no more than 20 words;

    context: {scenario}
    STORY:
    """

    prompt = PromptTemplate( input_variables=["scenario"], template=template)
    story_llm = LLMChain(llm=OpenAI(), prompt=prompt)
    story = story_llm.predict(scenario=scenario)
    return story


# text to speech
def text2speech(message):
    # API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_csmsc_conformer_fastspeech2"
    API_URL = "https://api-inference.huggingface.co/models/speechbrain/tts-tacotron2-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    return response.content    
    



def main():
    st.set_page_config(page_title="Image to audio story", page_icon="🤖")
    st.header('Change image into audio story')

    uploaded_file = st.file_uploader('Choose an image...', type='jpg')

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded image.', use_column_width=True)

        scenario = image2text(Image.open(uploaded_file))
        if scenario is not None:
            with st.expander("scenario"):
                st.write(scenario)

        story = generate_story(scenario)
        if story is not None:
            with st.expander("story"):
                st.write(story)

        audio = text2speech(story)

        if (audio is not None):
            if isinstance(audio, bytes) :
                if b'"error"' in audio:
                    st.error('text2speech model is spinning up, please try again in 5min')
                else:    
                    st.audio(audio, format="audio/flac")
            else:
                st.error('something wrong with text2speech model')    



if __name__ == '__main__':
    main()