import streamlit as st


def print_home():
    st.write("My Home")


def upload_image(container, img):
    container.image(img, width=700)


def build_header(header, img):
    header.title("Sound Of Failure")
    upload_image(header, img)


def build_body(body):
    body.header("An AI Solution to Detect Machine Failure")
    #body.header("Improving the cost of downtime of Industrial Machines")
    #body.write("We are working to build an AI solution"
    #           " to reduce industrial downtime using"
    #           " predictive maintenance. We are working towards"
    #           " a technology that is cheap, computationally inexpensive"
    #           " and easily deployable.")
               
    #body.subheader("The cost of industrial downtime")
    #body.write("Industries experience an average downtime of ~800 hours/year. The average cost of downtime can be as high as ~$20,000 per hour! Often a major cause of downtime is malfunctioning machines. During downtime, the overhead operating costs keeps growing without a significant increase in productivity. A survey in 2017 had found that 70% of companies cannot estimate when an equipment starts malfunctioning and only realise when it’s too late. If malfunctions can be detected early, downtime costs can be drastically reduced.")
    
    #body.subheader("Our solution")
    #body.write("We monitor the acoustic footprint of an industrial machine over time. We then use the recorded sound to diagnose early symptoms of malfunctions or machine failures. A machine will produce a different acoustic signature in its abnormal state compared to its normal state. Our AI algorithm can detect such different sound signatures with high degree of accuracy and can issue early warnings if indications of hardware malfunctions are detected.")
