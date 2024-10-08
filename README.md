## Disaster Tweet Classification Model

### Description

This project involves developing a machine learning model to classify tweets as indicating a disaster or not. Utilizing Deep Learning techniques, specifically a fine-tuned model from the Hugging Face library, the system is trained on the disaster tweet dataset from Kaggle. The goal is to predict whether a given tweet refers to a disaster event based on its content.


By analyzing critical components of tweets, such as content and context, the BERT model leverages its deep understanding of language to accurately classify whether a tweet indicates a disaster. The model is trained on a comprehensive dataset of disaster-related tweets, enabling it to effectively differentiate between disaster and non-disaster tweets across various contexts.

This classification system can be utilized by emergency responders, news organizations, and social media analysts to quickly identify and respond to disaster-related events or to monitor trends in disaster-related communications.


## Technologies Used

### Dataset

- **Source:** [Kaggle Disaster Tweets Dataset](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
- **Purpose:** Contains tweets labeled to indicate whether they refer to a disaster.

### Model

- **Base Model:** BERT (`bert-base-uncased`)
- **Library:** Hugging Face `transformers`
- **Task:** Binary text classification

### Approach

1. **Preprocessing:**
   - Load and preprocess the disaster tweet dataset.
   - Tokenize the tweet texts.

2. **Fine-Tuning:**
   - Fine-tune the BERT model on the preprocessed disaster tweet dataset.

3. **Training:**
   - Train the model to distinguish between disaster and non-disaster tweets.

### Key Technologies

- **Deep Learning (BERT):** For advanced text classification and contextual understanding.
- **Natural Language Processing (NLP):** For text preprocessing and analysis.
- **Machine Learning Algorithms:** For model training and prediction tasks.
- **Streamlit:** For creating an interactive web application interface.


## Google Colab Notebook

You can view and run the Google Colab notebook for this project [here](https://colab.research.google.com/drive/1Tl1lVcrGMyKZpwrqXKF7lxqL2444GFHo).



## Interfaces

### 1. Text Input Interface (`app.py`)

- **Description:** Allows users to paste tweet text and get a prediction of whether it indicates a disaster.
**Screenshots:**
  - **Actual Data (Disaster Tweet):**
 ![image](https://github.com/user-attachments/assets/27c42a5b-bd8c-44b8-95d0-e7ee4f0d7747)


- **Prediction Result:**
 ![image](https://github.com/user-attachments/assets/5cbd2fc2-ec4f-4c4a-ab78-48bd20ce3782)


- **Actual Data (Non-Disaster Tweet):**
  ![image](https://github.com/user-attachments/assets/277a312c-e7db-4719-9039-268013b427ec)


- **Prediction Result:**
![image](https://github.com/user-attachments/assets/39e8cbe5-8f5b-4168-aef3-afcf4c003531)




## Getting Started

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**
    - For `app.py`
      ```bash
      pip install -r requirements.txt
      ```

3. **Run the Applications**
    - For text input and classification (`app.py`):
      ```bash
      streamlit run app.py
      ```
    

4. **Provide a Tweet:** 
    - In `app.py`, paste the tweet text to classify.
    

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for transformer models.
- [NLTK](https://www.nltk.org/) for natural language processing.
- [Streamlit](https://streamlit.io/) for creating the interactive web interface.

## Author

[@AdilHayat](https://github.com/AdilHayat21173)

## Feedback

If you have any feedback, please reach out to us at [hayatadil300@gmail.com](mailto:hayatadil300@gmail.com).
