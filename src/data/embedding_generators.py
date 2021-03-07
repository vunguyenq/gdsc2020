import random
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence

EMBEDDING_SHAPE = 1024


class DataGeneratorFromEmbeddings(Sequence):
    """
    Takes a list of pairs and precomputed embeddings and labels
    and feeds them in batches to a deep learning model.
    """

    def __init__(
        self,
        pairs,
        embeddings,
        labels,
        batch_size=32,
        dim=EMBEDDING_SHAPE,
        shuffle=True,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.pairs = pairs
        self.embeddings = embeddings
        self.shuffle = shuffle
        self.indices = np.arange(len(self.pairs))
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
        indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # Find list of IDs
        pairs_temp = [self.pairs[k] for k in indices]
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
        if self.shuffle:
            np.random.shuffle(self.indices)

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
                x[0][i,] = self.embeddings[pair[0]]
                x[1][i,] = self.embeddings[pair[1]]
        else:
            # Generate data
            for i, pair in enumerate(pairs_temp):
                # Store sample
                x[0][i,] = self.embeddings[pair[0][0]]
                x[1][i,] = self.embeddings[pair[1][0]]
                # Store class
                y[i] = self.labels[pair]
        return x, y


class BalancedDataGeneratorFromEmbeddings(Sequence):
    """
    Reads classes from the train_data_path,
    removes the -1 class and classes with one picture and
    creates batches containing a balanced amount of matching
    and non-matching samples.
    """

    def __init__(
        self,
        train_data_path,
        embeddings,
        batch_size=32,
        dim=EMBEDDING_SHAPE,
        shuffle=True,
    ):
        assert batch_size % 2 == 0, "Batch size must be an even number"
        self.train_data_path = Path(train_data_path)
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.id_to_images, self.image_to_id = self.__parse_folder()
        self.indices = list(self.image_to_id.keys())
        self.on_epoch_end()
        print('BalancedDataGeneratorFromEmbeddings created!')

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        half_batch = self.batch_size / 2
        return int(np.floor(len(self.indices) / half_batch))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        # Generate indexes of the batch
        half_batch = int(self.batch_size / 2)
        start_index = index * half_batch
        end_index = start_index + half_batch
        images = self.indices[start_index:end_index]
        x, y = self.__data_generation(images)
        return x, y

    def get_sample(self, index):
        """
        Gets one sample batch
        :param index:
        :return:
        """
        x, y = self.__getitem__(index)
        print(f'get_sample() called, index = {index}')
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __parse_folder(self):
        """
        Creates id_to_images and image_to_ids mappings.
        Ignores all whale ids with only one image and the -1 class
        :return:
        """
        id_folders = list(self.train_data_path.glob("*"))
        id_to_images = {}
        image_to_id = {}
        for folder in id_folders:
            if folder.name == "-1":
                continue
            pics = list(folder.glob("*.jpg"))
            if len(pics) == 1:
                continue
            id_to_images[folder.name] = {p.name for p in pics}
            for p in pics:
                image_to_id[p.name] = folder.name
        return id_to_images, image_to_id

    def __data_generation(self, images):
        """
        Generates data containing batch_size samples
        :param images:
        :return:
        """
        # Initialization
        x = [np.empty((self.batch_size, self.dim)) for _i in range(2)]
        y = np.empty(self.batch_size, dtype=int)
        # Generate data
        for i, image in enumerate(images):
            output_index = 2 * i

            # Store similar sample
            similar_img = self.__get_similar_image(image)
            x[0][output_index,] = self.embeddings[image]
            x[1][output_index,] = self.embeddings[similar_img]
            y[output_index] = 1

            # Store different sample
            different_img = self.__get_different_image(image)
            x[0][output_index + 1,] = self.embeddings[image]
            x[1][output_index + 1,] = self.embeddings[different_img]
            y[output_index + 1] = 0

        return x, y

    def __get_similar_image(self, image):
        """
        Returns an image of the same class than input image
        :param image: img file name to search for similar one
        :return: file name of similar image
        """
        image_id = self.image_to_id[image]
        potential_images = self.id_to_images[image_id].difference(set([image]))
        similar_image = random.sample(potential_images, 1)[0]
        return similar_image

    def __get_different_image(self, image):
        """
        Returns and image of a different class than input class
        :param image: class to exclude
        :return: file name for an image not in class img_id
        """
        image_id = self.image_to_id[image]
        while True:
            candidate = random.sample(self.indices, 1)[0]
            if self.image_to_id[candidate] != image_id:
                return candidate

# Improvement of BalancedDataGeneratorFromEmbeddings: cover all similar pairs (7044) 
class BalancedDataGeneratorFromEmbeddings2(Sequence):
    """
    Reads classes from the train_data_path,
    removes the -1 class and classes with one picture and
    creates batches containing a balanced amount of matching
    and non-matching samples.
    get_similar_image and get_different_image will NOT take random images
    """

    def __init__(
        self,
        train_data_path,
        embeddings,
        batch_size=32,
        dim=EMBEDDING_SHAPE,
        shuffle=True,
    ):
        assert batch_size % 2 == 0, "Batch size must be an even number"
        self.train_data_path = Path(train_data_path)
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.id_to_images, self.image_to_id = self.__parse_folder()
        self.indices = list(self.image_to_id.keys())
        self.on_epoch_end()
        print('BalancedDataGeneratorFromEmbeddings created!')

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        half_batch = self.batch_size / 2
        return int(np.floor(len(self.indices) / half_batch))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        # Generate indexes of the batch
        half_batch = int(self.batch_size / 2)
        start_index = index * half_batch
        end_index = start_index + half_batch
        images = self.indices[start_index:end_index]
        x, y = self.__data_generation(images)
        return x, y

    def get_sample(self, index):
        """
        Gets one sample batch
        :param index:
        :return:
        """
        x, y = self.__getitem__(index)
        print(f'get_sample() called, index = {index}')
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __parse_folder(self):
        """
        Creates id_to_images and image_to_ids mappings.
        Ignores all whale ids with only one image and the -1 class
        :return:
        """
        id_folders = list(self.train_data_path.glob("*"))
        id_to_images = {}
        image_to_id = {}
        for folder in id_folders:
            if folder.name == "-1":
                continue
            pics = list(folder.glob("*.jpg"))
            if len(pics) == 1:
                continue
            id_to_images[folder.name] = {p.name for p in pics}
            for p in pics:
                image_to_id[p.name] = folder.name
        return id_to_images, image_to_id

    def __data_generation(self, images):
        """
        Generates data containing batch_size samples
        :param images:
        :return:
        """
        # Initialization
        x = [np.empty((self.batch_size, self.dim)) for _i in range(2)]
        y = np.empty(self.batch_size, dtype=int)
        # Generate data
        for i, image in enumerate(images):
            output_index = 2 * i

            # Store similar sample
            similar_img = self.__get_similar_image(image)
            x[0][output_index,] = self.embeddings[image]
            x[1][output_index,] = self.embeddings[similar_img]
            y[output_index] = 1

            # Store different sample
            different_img = self.__get_different_image(image)
            x[0][output_index + 1,] = self.embeddings[image]
            x[1][output_index + 1,] = self.embeddings[different_img]
            y[output_index + 1] = 0

        return x, y

    def __get_similar_image(self, image):
        """
        Returns an image of the same class than input image
        :param image: img file name to search for similar one
        :return: file name of similar image
        """
        image_id = self.image_to_id[image]
        potential_images = self.id_to_images[image_id].difference(set([image]))
        similar_image = random.sample(potential_images, 1)[0]
        return similar_image

    def __get_different_image(self, image):
        """
        Returns and image of a different class than input class
        :param image: class to exclude
        :return: file name for an image not in class img_id
        """
        image_id = self.image_to_id[image]
        while True:
            candidate = random.sample(self.indices, 1)[0]
            if self.image_to_id[candidate] != image_id:
                return candidate
