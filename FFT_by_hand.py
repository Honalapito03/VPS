import numpy as np


def recursive_fft(x):
    """
    Compute the FFT of a 1D array x using recursion.
    Length of x must be a power of 2.
    """
    N = len(x)
    if N <= 1:
        return x
    else:
        # FFT of even and odd indexed elements
        X_even = recursive_fft(x[::2])
        X_odd  = recursive_fft(x[1::2])
        
        # Combine
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        #print("factor:", len(factor), len(X_odd), len(X_even))
        #print(factor[:N//2], factor[N//2:])
        return np.concatenate([X_even + factor[:N//2] * X_odd,
                               X_even + factor[N//2:] * X_odd])

def recursive_ifft(X):
    """
    Compute the IFFT of a 1D array X using recursion.
    Length of X must be a power of 2.
    """
    N = len(X)
    if N <= 1:
        return X
    else:
        x_even = recursive_ifft(X[::2])
        x_odd  = recursive_ifft(X[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return (np.concatenate([x_even + factor[:N//2] * x_odd,
                                x_even + factor[N//2:] * x_odd])) / 2
    
def fft2d(matrix):
    # FFT along rows
    row_transformed = np.array([recursive_fft(row) for row in matrix])
    # FFT along columns (apply FFT to transpose, then transpose back)
    col_transformed = np.array([recursive_fft(col) for col in row_transformed.T]).T
    return col_transformed

def ifft2d(matrix):
    # IFFT along rows
    row_transformed = np.array([recursive_ifft(row) for row in matrix])
    # IFFT along columns
    col_transformed = np.array([recursive_ifft(col) for col in row_transformed.T]).T
    return col_transformed



# Example usage:
if __name__ == "__main__":
    # Create a sample 2D array (4x4 for simplicity)
    sample_matrix = np.array([[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]], dtype=float)

    print("Original Matrix:")
    print(sample_matrix)

    # Compute FFT
    fft_result = fft2d(sample_matrix)
    print("\nFFT Result:")
    print(fft_result)

    # Compute IFFT
    ifft_result = ifft2d(fft_result)
    print("\nReconstructed Matrix after IFFT:")
    print(ifft_result.real)
