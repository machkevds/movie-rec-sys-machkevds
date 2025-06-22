# Recommendation System for Movie Ratings (MovieLens 100K)

This project uses MovieLens 100K Dataset to create 4 differet collaborative filtering models to make user predicitons on movie preferences, including a web app to showcase such models to make predictions, the models are:

- User-User Cosine
- User-User Pearson
- Item-Item Cosine
- Item-Item Pearson


---

## Run the App in Google Colab

You can run the Streamlit app directly from Google Colab:

## 1. Upload the following to your Colab session:
- `app_movie_recsys.py` — the Streamlit app.
- `user_item_matrix.pkl`
- `user_sim_cosine.pkl`
- `user_sim_pearson`
- `user_item_centered`
- `user_means`
- `item_similarity_cosine`
- `item_similarity_pearson`
- `item_means`
- `movies_df`

## 2. Run the following setup:
### - Install dependencies
!pip install streamlit

### - Fetch tunnel IP for deployment
!wget -q -O - ipv4.icanhazip.com

## - Launch the Streamlit app and create public link
!streamlit run `app_name.py` & npx localtunnel --port 8501
'''

## EXTRA NOTES:
### - When prompted, you may see the following, type y and hit enter.

Need to install the following packages:
localtunnel@2.0.2
Ok to proceed? (y)

### - You will be asked to confirm tunnel IP with a Prompt
E.g. 'XX.XXX.XX.XX'


### - The link will be printed in the output cell. Example:
#### Your app is live at: 'https://silent-example-go.loca.lt'
