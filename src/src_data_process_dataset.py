import numpy as np
from tensorflow.keras.utils import Sequence


class DataGeneratorFromEmbeddings(Sequence):
    """
    Generates data for our model
    """
    def __init__(self, pairs, embeddings, labels, batch_size=32, dim=1024, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.pairs = pairs
        self.embeddings = embeddings
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        pairs_temp = [self.pairs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(pairs_temp)
        return x, y

    def get_sample(self, index):
        """
        Gets one sample batch
        :param index:
        :return:
        """
        x, y = self.__getitem__(index)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, pairs_temp):
        """
        Generates data containing batch_size samples
        :param pairs_temp:
        :return:
        """
        # Initialization
        x = [np.empty((self.batch_size, self.dim)) for i in range(2)]
        y = np.empty(self.batch_size, dtype=int)
        if self.labels is None:
            for i, pair in enumerate(pairs_temp):
                x[0][i, ] = self.embeddings[pair[0]]
                x[1][i, ] = self.embeddings[pair[1]]
            return x, y
        else:
            # Generate data
            for i, pair in enumerate(pairs_temp):
                # Store sample
                x[0][i, ] = self.embeddings[pair[0][0]]
                x[1][i, ] = self.embeddings[pair[1][0]]
                # Store class
                y[i] = self.labels[pair]
            return x, y
