import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit as st
from PIL import Image
import torch
import os


def app():
    with open('apps/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        #Assign variable
    model = torch.hub.load(os.getcwd()+ '/yolov5', 'custom', path=r'streamlit-multiapps/yolov5/playing_cards.pt', source='local',force_reload= True)
    def process(image):
            image = cv2.flip(image, 1)
            image = run2Die(img = image)
            return image
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processImg = run2Die(Image.fromarray(img))
        cv2.flip(processImg)
        return av.VideoFrame.from_ndarray(processImg.render(),format="bgr24")

    def run2Die(
        isCamera = False,
        img = Image.open('a1.jpg')
    ):
        numpydata = np.asarray(img)
        
        res = model(numpydata)
        #res.show()
        return res 
    #UI elements
    st.title('Home')
    col1, col2= st.columns(2)
    with col1:
        st.header("Webcam")
        # uploaded_file = st.file_uploader("Choose a file")
        # if uploaded_file is not None:
        #     image = Image.open(uploaded_file.name)
        #     processedimg = run2Die(im = image)
        #     st.image(image, caption='Sunrise by the mountains')
        #     processedimg.show()
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        webrtc_ctx = webrtc_streamer(key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}, 
        video_frame_callback=video_frame_callback,
        async_processing= False)
    with col2:
        a = st.container()
        a.header("A dog")
        a.uploaded_file = st.file_uploader("Choose a image", type=['jpg','png'])
        if a.uploaded_file is not None:
            image = Image.open(a.uploaded_file)
            processedimg = run2Die(img = image)
            st.image(processedimg.render(), caption='')
        b = st.container()
        b.uploaded_file = st.file_uploader("Choose a video", type = ['mp4'])
        if b.uploaded_file is not None:
            stream = b.uploaded_file.getvalue()
            
            ret,frame = stream.read()
            if ret == True:
                processedimg = run2Die(img = frame)
                video = av.VideoFrame.from_ndarray(np.asarray(processedimg.render()[:,:,:1]),format="bgr24")
                st.video(video,format='video/mp4',start_time=0)
