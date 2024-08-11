# Veersa-vanguard:Adverse Event Prediction Based on Symptoms

This project predicts diseases based on user-input symptoms using a machine learning model. Users can manually input symptoms or upload an audio file for transcription and analysis.

## Features

- **Symptom Input**: Users can input symptoms manually via text or upload an audio file.
- **Audio Transcription**: The app transcribes audio files to text and extracts symptoms.
- **Symptom Matching**: The app matches user symptoms with a dataset of symptoms and diseases.
- **Disease Prediction**: Based on the input symptoms, the app predicts the most likely diseases.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or above installed.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/disease-prediction.git
    cd disease-prediction
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary data files and the pre-trained model:
   - `dis_sym_dataset_comb.csv`
   - `dis_sym_dataset_norm.csv`
   - `linear_regression_model.pkl`

   Place these files in the project directory.

### Running the Application

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Access the application in your web browser at `http://localhost:8501`.

### Usage

- **Manual Input**: Enter symptoms separated by commas in the provided text area.
- **Audio Input**: Upload an audio file (`.mp3`, `.wav`, or `.m4a`), and the app will transcribe it to text.

After providing the input, the app will:
- Match symptoms with a dataset.
- Suggest additional symptoms based on co-occurrence with selected symptoms.
- Predict the top 10 diseases based on the input symptoms.

### Project Structure

- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `README.md`: This documentation file.
- `dis_sym_dataset_comb.csv`: Combined symptom dataset.
- `dis_sym_dataset_norm.csv`: Normalized symptom dataset.
- `linear_regression_model.pkl`: Pre-trained machine learning model.

### Dependencies

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- BeautifulSoup4
- Requests

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

### Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

### Contact

For questions or suggestions, feel free to reach out to [tanmay13tanwar@gmail.com](mailto:tanmay13tanwar@gmail.com).
