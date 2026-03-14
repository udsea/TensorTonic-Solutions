import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    L = max_len if max_len is not None else (max((len(seq) for seq in seqs), default=0))

    output = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        seq = np.asarray(seq)
        length = min(len(seq), L)
        output[i, :length] = seq[:length]

    return output