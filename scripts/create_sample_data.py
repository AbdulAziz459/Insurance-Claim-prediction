import pandas as pd
import numpy as np

def create_porto_seguro_sample():
    """Create a sample dataset similar to Porto Seguro structure"""
    np.random.seed(42)
    n_samples = 10000
    
    # Create the data dictionary
    data = {
        'id': range(n_samples),
    }
    
    # Individual features
    data['ps_ind_01'] = np.random.randint(0, 6, n_samples)
    data['ps_ind_02_cat'] = np.random.randint(1, 5, n_samples)
    data['ps_ind_03'] = np.random.randint(0, 12, n_samples)
    data['ps_ind_04_cat'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_05_cat'] = np.random.choice([0, 1, 2, 4, 6, -1], n_samples, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    data['ps_ind_06_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_07_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_08_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_09_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_10_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_11_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_12_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_13_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_14'] = np.random.randint(0, 15, n_samples)
    data['ps_ind_15'] = np.random.randint(0, 15, n_samples)
    data['ps_ind_16_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_17_bin'] = np.random.randint(0, 2, n_samples)
    data['ps_ind_18_bin'] = np.random.randint(0, 2, n_samples)
    
    # Regional features
    data['ps_reg_01'] = np.random.uniform(0, 1, n_samples)
    data['ps_reg_02'] = np.random.uniform(0, 2, n_samples)
    data['ps_reg_03'] = np.random.choice([-1] + list(np.random.uniform(0, 3, 100)), n_samples)
    
    # Car features
    data['ps_car_01_cat'] = np.random.randint(6, 12, n_samples)
    data['ps_car_02_cat'] = np.random.choice([0, 1, -1], n_samples, p=[0.4, 0.5, 0.1])
    data['ps_car_03_cat'] = np.random.choice([-1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.3, 0.3])
    data['ps_car_04_cat'] = np.random.randint(0, 10, n_samples)
    data['ps_car_05_cat'] = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.4, 0.4])
    data['ps_car_06_cat'] = np.random.randint(0, 18, n_samples)
    data['ps_car_07_cat'] = np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.4, 0.5])
    data['ps_car_08_cat'] = np.random.randint(0, 2, n_samples)
    data['ps_car_09_cat'] = np.random.randint(0, 6, n_samples)
    data['ps_car_10_cat'] = np.random.randint(0, 3, n_samples)
    data['ps_car_11_cat'] = np.random.randint(0, 105, n_samples)
    data['ps_car_11'] = np.random.randint(0, 105, n_samples)
    data['ps_car_12'] = np.random.uniform(0, 1, n_samples)
    data['ps_car_13'] = np.random.uniform(0, 4, n_samples)
    data['ps_car_14'] = np.random.uniform(-1, 1, n_samples)
    data['ps_car_15'] = np.random.uniform(0, 4, n_samples)
    
    # Calculated features
    for i in range(1, 15):
        data[f'ps_calc_{i:02d}'] = np.random.uniform(0, 1, n_samples)
    
    for i in range(15, 21):
        data[f'ps_calc_{i}_bin'] = np.random.randint(0, 2, n_samples)
    
    # Target variable with realistic class imbalance
    data['target'] = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save training data
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index).drop('target', axis=1)
    
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print(f"Created train.csv with {len(train_df)} rows")
    print(f"Created test.csv with {len(test_df)} rows")
    print(f"Target distribution in training data:")
    print(train_df['target'].value_counts())

if __name__ == "__main__":
    create_porto_seguro_sample()
