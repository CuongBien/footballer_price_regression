"""
Log Transform Wrapper cho Target Variable
Giúp cải thiện performance với target có range lớn
"""
import numpy as np
import pickle


class LogTargetTransformer:
    """
    Transform target variable sang log scale để giảm impact của outliers
    và cải thiện model performance với data có range lớn
    """
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, y):
        """Fit transformer (không làm gì, chỉ để tương thích API)"""
        self.is_fitted = True
        return self
    
    def transform(self, y):
        """
        Transform y sang log scale
        Dùng log1p để handle giá trị 0
        """
        return np.log1p(y)
    
    def inverse_transform(self, y_log):
        """
        Transform predictions từ log scale về scale gốc
        """
        return np.expm1(y_log)
    
    def fit_transform(self, y):
        """Fit và transform"""
        self.fit(y)
        return self.transform(y)
    
    def save(self, filepath):
        """Save transformer"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load transformer"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Test
    transformer = LogTargetTransformer()
    
    # Test data
    y = np.array([100000, 1000000, 10000000, 100000000])
    
    print("Original values:", y)
    print("Log transformed:", transformer.fit_transform(y))
    print("Inverse transform:", transformer.inverse_transform(transformer.transform(y)))
    
    # Verify
    assert np.allclose(y, transformer.inverse_transform(transformer.transform(y)))
    print("\n✓ Log transform working correctly!")
