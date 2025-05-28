from model import FraudDetectionModel
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv')
    
    print("Initializing model...")
    model = FraudDetectionModel()
    
    print("Preprocessing data...")
    X, y = model.preprocess_data(df)
    X_balanced, y_balanced = model.balance_data(X, y)
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42
    )
    
    print("Building model...")
    model.build_model(input_shape=(1, 30))
    
    print("Training model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    print("Saving model...")
    model.save_model()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()