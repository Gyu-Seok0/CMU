import numpy as np
import pickle
from pyturbo import Stage


class BagOfWords(Stage):

    """
    Input: features [N x D]
    Output: bag-of-words [W]
    """

    def allocate_resource(self, resources, *, weight_path):
        self.weight_path = weight_path
        self.weight = None
        return [resources]

    def reset(self):
        if self.weight is None:
            with open(self.weight_path, 'rb') as f:
                self.weight = pickle.load(f)

    def get_bag_of_words(self, features: np.ndarray) -> np.ndarray:
        """
        features: [N x D] -> 32*128

        Return: count of each word, [W] -> cluster num?
        """
        # TODO: Generate bag of words
        # Calculate pairwise distance between each feature and each cluster,
        # assign each feature to the nearest cluster, and count
        #print("self.weight",self.weight)
        labels = [0] * self.weight.shape[0]

        for f in features:
            f = f.reshape(1,-1) # (1, 128)
            distance = (self.weight - f)**2 # (10,128)
            distance2 = np.sum(distance, axis = 1)
            label = np.argmin(distance2)
            labels[label] += 1
        
        return np.array(labels)

    def get_video_feature(self, bags: np.ndarray) -> np.ndarray:
        """
        bags: [B x W]

        Return: pooled vector, [W]
        """
        # TODO: Aggregate frame-level bags into a video-level feature.
        return np.max(bags, axis = 0)

    def process(self, task):
        features = task.content
        bags = []
        for frame_features in features:
            bag = self.get_bag_of_words(frame_features)
            assert isinstance(bag, np.ndarray)
            assert bag.shape == self.weight.shape[:1]
            bags.append(bag)
        bags = np.stack(bags)
        video_bag = self.get_video_feature(bags)
        assert isinstance(video_bag, np.ndarray)
        assert video_bag.shape == self.weight.shape[:1]
        return task.finish(video_bag)
