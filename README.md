# Movie-Review-classifier-

This project analyzes the sentiment of film reviews from the IMDB dataset. Using recurrent neural networks (RNNs) with LSTM layers, the model classifies reviews as positive or negative. The project involves data preprocessing and training a binary classification model to accurately predict the sentiment of new reviews.

The dataset can be found on Kaggle https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews. 

In this project, model paths refer to the locations where trained models are saved and loaded. Here's a simple explanation for beginners:

Training and Saving the Model:
After training the model with a dataset, we want to keep the trained model's parameters for future predictions.
To do this, we save the model's parameters (also known as the model's state dictionary) to a file using torch.save().
This file is placed in a location of your choice, known as the model path. For instance, we might save it as sentiment_model.pth at a particular directory like /path/to/your/model.

Loading and Using the Model:
Once saved, we can load the model later for making predictions or further training.
First, we re-create the model architecture with the same parameters used during training.
Then, we use torch.load() to read the saved parameters from the model path and apply them to the newly created model object using .load_state_dict().
The model is now ready for real-time predictions using new data, ensuring consistent behavior with the training phase.

---------------------------------------------------------------------------------------------------------------------------------------------

Key Points for Beginners:

Understanding TF-IDF Vectorizer:
What It Does: Converts text data into numerical vectors by calculating the importance of words (terms) relative to the document and the entire dataset.
Why It's Used: Helps the model understand and process text data by transforming words into numbers.

Using PyTorch Tensors:
What They Are: Multi-dimensional arrays that store data for processing in neural networks.
Why They're Important: Tensors are the fundamental building blocks in PyTorch, allowing efficient data handling and computation.

DataLoader Utility:
What It Does: Handles data in batches, making it more manageable and faster for the model to process during training.
Why It's Used: Improves training efficiency by breaking the data into smaller, sequential chunks.

LSTM Model Architecture:
What It Is: A type of Recurrent Neural Network (RNN) that is particularly good at learning from sequences, like sentences or time series data.
Why It's Used: Captures the dependencies and relationships in sequential data, which is crucial for understanding the context in text.

Saving and Loading Models:
Saving: torch.save() saves the model’s parameters to a file.
Loading: torch.load() loads the parameters into a model architecture for reuse.
Why This Matters: Ensures you don’t have to retrain the model from scratch every time you want to use it.

Clean Review Function:
What It Does: Processes and prepares text data by removing unnecessary elements (like HTML tags, punctuation) and converting text to a standard form.
Why It's Important: Standardizes the input data, making it easier for the model to learn and make accurate predictions.

Tips for Beginners:

Start Simple: Focus on understanding each part of the process step by step.
Read Documentation: Refer to library documentation (like PyTorch or Scikit-learn) for detailed explanations and examples.
Experiment: Try modifying parts of the script to see how changes affect the outcome.
Ask for Help: You can reach me on discord @sleepytimebaby. 

---------------------------------------------------------------------------------------------------------------------------------------------

Step 1: Preprocessing the IMDB Dataset

In this step, we clean, lemmatize, and split the dataset into training and testing sets for later use in the sentiment analysis model.

Import Libraries:
        NLTK: Natural language processing tools, including stop words and lemmatization.
        Pandas: For efficient data manipulation and reading/writing CSV files.
        Re and String: For text preprocessing and cleaning.
        Scikit-learn's Train-Test Split: For splitting data into training and testing sets.

    import nltk

    nltk.download('stopwords')
    nltk.download('wordnet')

    import pandas as pd
    import re
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.model_selection import train_test_split

Load the Dataset:

Read the CSV file containing the IMDB movie reviews into a DataFrame.

    file_path = '/PATH/TO/YOUR/DATA/'
    df = pd.read_csv(file_path)

Initialize Preprocessing Tools:

Stop Words: A set of common words to be removed.
Lemmatizer: For reducing words to their root forms.

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

Define Review Cleaning Function:

This function removes HTML tags, punctuation, and digits, converts to lowercase, filters out stop words, and lemmatizes the remaining tokens.

    def clean_review(text):
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = text.lower()  # Convert to lowercase
        text = re.sub(f"[{string.punctuation}\d]", "", text)  # Remove punctuation and digits
        tokens = [word for word in text.split() if word not in stop_words]  # Remove stop words
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
        return " ".join(tokens)

Apply Cleaning Function to Reviews:

Use the cleaning function to process all reviews and store the results in a new column.

    df['clean_review'] = df['review'].apply(clean_review)

Map Sentiments to Numerical Values:

Convert "positive" and "negative" sentiments to 1 and 0, respectively, for compatibility with machine learning models.

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

Split Dataset into Training and Testing Sets:

Create an 80-20 split of the data for training and evaluation.

    X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)

Display the sizes of the training and test sets.

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

Save Preprocessed Data:

Save the cleaned and split data into CSV files for model training and testing.

    train_data = pd.DataFrame({'review': X_train, 'sentiment': y_train})
    test_data = pd.DataFrame({'review': X_test, 'sentiment': y_test})

    train_data.to_csv('IMDB_train_preprocessed.csv', index=False)
    test_data.to_csv('IMDB_test_preprocessed.csv', index=False)

--------------------------------------------------------------------------------------------------------------------------------------------

Step 2: Feature Extraction and Model Training

This step involves converting text data into numerical features using TF-IDF vectorization, training an LSTM model to classify sentiments, and saving the trained model for future use. Below is a detailed breakdown of the script.

Import Necessary Libraries:

Pandas: For data manipulation and loading CSV files.
Sklearn's TfidfVectorizer: For converting text data to TF-IDF features.
Torch: For creating and training the neural network.
Pickle: For saving the TF-IDF vectorizer.

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pickle

Load Preprocessed Data:

The script reads the training and test datasets from CSV files.

    train_data = pd.read_csv('/PATH/TO/YOUR/TRAINING/DATA')
    test_data = pd.read_csv('/PATH/TO/YOUR/TEST/DATA')

Feature Extraction Using TF-IDF:

Convert the text reviews into TF-IDF vectors.
Limit the number of features to 5000 for simplicity and efficiency.
Save the vectorizer to a file for future use.

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['review']).toarray()
    X_test = vectorizer.transform(test_data['review']).toarray()

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

Convert Data to PyTorch Tensors:

Transform the TF-IDF vectors and sentiment labels into PyTorch tensors for compatibility with the neural network.

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_data['sentiment'].values, dtype=torch.long)
    y_test_tensor = torch.tensor(test_data['sentiment'].values, dtype=torch.long)

Create DataLoader for Batching:

Organize data into batches using DataLoader to optimize training efficiency.

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

Define an LSTM Model:

The LSTM model processes sequences and outputs sentiment predictions.
The model consists of an LSTM layer followed by a fully connected layer.

    class SentimentLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SentimentLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), 100)
            c0 = torch.zeros(1, x.size(0), 100)
            out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

Initialize Model, Loss Function, and Optimizer:

Define the model's input size, hidden size, and number of output classes.
Use CrossEntropyLoss for classification and Adam optimizer for training.

    input_size = 5000
    hidden_size = 100
    num_classes = 2
    model = SentimentLSTM(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

Training Loop:

Train the model for 20 epochs, adjusting weights to minimize loss.


    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

Evaluation:

Test the model on the test dataset and calculate accuracy.

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

Save the Trained Model:

Save the trained model's state dictionary for later use.

    torch.save(model.state_dict(), '/PATH/TO/YOUR/MODEL')

--------------------------------------------------------------------------------------------------------------------------------------------

Step 3: Model Initialization and Real-Time Classification

This step involves loading the pre-trained LSTM model and the TF-IDF vectorizer, cleaning user-input reviews, transforming them into numerical vectors, and predicting the sentiment. Below is the detailed explanation of the script.

Import Necessary Libraries:

Torch: For loading the pre-trained model and handling tensor operations.
Pickle: For loading the saved TF-IDF vectorizer.
Sklearn's TfidfVectorizer: For transforming text data to TF-IDF features.
Re and String: For text cleaning and preprocessing.

    import torch
    import torch.nn as nn
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re
    import string

Load the TF-IDF Vectorizer:

Load the pre-saved TF-IDF vectorizer to transform new input reviews consistently with the training data.

    with open('/PATH/TO/YOUR/VECTORIZER', 'rb') as f:
        vectorizer = pickle.load(f)

Define the LSTM Model Class:

The model architecture should match the one used during training to ensure compatibility.

    class SentimentLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SentimentLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), 100)
            c0 = torch.zeros(1, x.size(0), 100)
            out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


Initialize the Model:

Define the model with the same parameters used during training.
Load the saved model state to ensure it can make accurate predictions.

    input_size = 5000  # Size of TF-IDF vectors
    hidden_size = 100
    num_classes = 2
    model = SentimentLSTM(input_size, hidden_size, num_classes)

    model_path = 'PATH/TO/YOUR/MODEL'
    model.load_state_dict(torch.load(model_path))
    model.eval()

Function to Clean and Preprocess Input Review:

This function prepares the user-provided review by removing HTML tags, converting text to lowercase, and stripping out punctuation and digits.

    def clean_review(text):
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = text.lower()  # Convert to lowercase
        text = re.sub(f"[{string.punctuation}\d]", "", text)  # Remove punctuation and digits
        tokens = text.split()  # Tokenize
        return " ".join(tokens)

Function to Classify User Input Review:

This function takes a user-provided review, cleans it, transforms it into a TF-IDF vector, converts it to a tensor, and uses the model to predict sentiment.

    def classify_review():
        user_review = input("Enter a movie review: ")
        cleaned_review = clean_review(user_review)
        review_vector = vectorizer.transform([cleaned_review]).toarray()
        review_tensor = torch.tensor(review_vector, dtype=torch.float32)

        with torch.no_grad():
            output = model(review_tensor)
            _, predicted = torch.max(output, 1)
            sentiment = "You enjoyed the movie." if predicted.item() == 1 else "You did not like the movie."
            print(f"Sentiment: {sentiment}")

Start the Classification Process:

Initiate the function to classify the input review and display the sentiment result.

    classify_review()



