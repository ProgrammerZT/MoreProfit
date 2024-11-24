import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self, a500_path, c1000_path, fixed_bench_mark_return=None):
        """
        Initializes the DataPreprocessor with file paths for the datasets.

        Parameters:
        a500_path (str): Path to the A500 dataset CSV file.
        c1000_path (str): Path to the C1000 dataset CSV file.
        fixed_bench_mark_return (float):fixed daily bench_mark_return
        """
        self.a500_data = pd.read_csv(a500_path)
        self.c1000_data = pd.read_csv(c1000_path)
        self.merged_data = None
        self.features = None
        self.targets = None
        self.benchmark_returns = fixed_bench_mark_return

    def preprocess_data(self):
        """
        Process the raw data to generate features, targets, and benchmark returns.

        Returns:
        tuple: A tuple containing the features, targets, and benchmark returns.
        """
        # Normalize the '涨跌幅' columns
        self.a500_data['涨跌幅'] = self.a500_data['涨跌幅'] / 100
        self.c1000_data['涨跌幅'] = self.c1000_data['涨跌幅'] / 100

        # Merge A500 and C1000 data on '日期'
        self.merged_data = pd.merge(self.a500_data, self.c1000_data, on='日期', suffixes=('_A500', '_C1000'))

        # Extract features
        self.features = self.merged_data[['开盘_A500',
                                          '收盘_A500',
                                          '最高_A500',
                                          '最低_A500',
                                          '成交量_A500',
                                          '成交额_A500',
                                          '振幅_A500',
                                          '涨跌幅_A500',
                                          '涨跌额_A500',
                                          '换手率_A500',
                                          '开盘_C1000',
                                          '收盘_C1000',
                                          '最高_C1000',
                                          '最低_C1000',
                                          '成交量_C1000',
                                          '成交额_C1000',
                                          '振幅_C1000',
                                          '涨跌幅_C1000',
                                          '涨跌额_C1000',
                                          '换手率_C1000']].values

        # Extract targets (涨跌幅 for A500 and C1000)
        self.targets = self.merged_data[['涨跌幅_A500', '涨跌幅_C1000']].values

        if self.benchmark_returns is not None:
            # Set benchmark returns to the fixed value (5% annualized return, i.e., 0.05 / 250 per day)
            self.benchmark_returns = np.full(self.merged_data.shape[0], self.benchmark_returns / 250)

        else:
            # Benchmark returns (涨跌幅_A500)
            self.benchmark_returns = self.merged_data['涨跌幅_A500'].values

        # Ensure consistency of sizes
        assert self.features.shape[0] == self.targets.shape[0] == self.benchmark_returns.shape[
            0], "Data length mismatch!"

        return self.features, self.targets, self.benchmark_returns


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor with dataset paths
    preprocessor = DataPreprocessor('data/A500_train.csv', 'data/C1000_train.csv')

    # Preprocess data
    features, targets, benchmark_returns = preprocessor.preprocess_data()

    # Print shapes to verify correctness
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Benchmark returns shape: {benchmark_returns.shape}")
