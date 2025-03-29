import streamlit as st

st.title("This is a Title")
st.header("This is a Header")
st.subheader("This is a Subheader")
st.text("This is simple text")
st.write("This is formatted text")


user_text = st.text_input("Enter your name:")
st.write("Hello,", user_text)

if st.button("Click Me"):
    st.write("You clicked the button!")


number = st.slider("Pick a number", 0, 100, 50)
st.write("You selected:", number)


choice = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", choice)
