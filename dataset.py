import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Label encoding: 1 for significant rise, 0 for significant drop
    data['label'] = data['Pct_Change'].apply(lambda x: 1 if x >= 10 else 0)

    # Split into training and temp sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        data['Headline'], data['label'], test_size=0.2, random_state=42
    )

    # Split temp into validation and test sets
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
