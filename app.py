import streamlit as st 
import joblib,os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image

# load Vectorizer For Gender Prediction
news_vectorizer = open("models/final_news_cv_vectorizer.pkl","rb")
news_cv = joblib.load(news_vectorizer)

# # load Model For Gender Prediction
# news_nv_model = open("models/naivebayesgendermodel.pkl","rb")
# news_clf = joblib.load(news_nv_model)

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model



# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def main():

	st.title("News Classifier")

	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;"> Want to get your news classified? </h1>
	</div>

	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Enter the news and get it classified")
		news_text = st.text_area("Enter News Here","Type Here")
		all_ml_models = ["NB"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'Business': 0,'Science and Technology': 1,'Sport': 2,'Health': 3,'Politics': 4,'Entertainment & Lifestyle': 5}
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(news_text))
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'NB':
				predictor = load_prediction_models("models/newsclassifier_Logit_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)




			final_result = get_key(prediction,prediction_labels)
			st.success("News Categorized as:: {}".format(final_result))



if __name__ == '__main__':
	main()

