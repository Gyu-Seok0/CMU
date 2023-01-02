import numpy as np
import torch
from pyturbo import Stage
from torch.backends import cudnn
from torchvision.models import video as video_models
from torchvision.models.feature_extraction import create_feature_extractor


class CNN3DFeature(Stage):

    """
    Input: a clip [T x H x W x C]
    Output: CNN feature [D]
    """

    def allocate_resource(self, resources, *, model_name, weight_name,
                          node_name, replica_per_gpu=1):
        self.model_name = model_name
        self.weight_name = weight_name
        self.node_name = node_name
        self.model = None
        gpus = resources.get('gpu')
        self.num_gpus = len(gpus)
        if len(gpus) > 0:
            return resources.split(len(gpus)) * replica_per_gpu
        return [resources]

    def reset(self):
        if self.model is None:
            gpu_ids = self.current_resource.get('gpu', 1)
            if len(gpu_ids) >= 1:
                self.device = 'cuda:%d' % (gpu_ids[0])
                cudnn.fastest = True
                cudnn.benchmark = True
            else:
                self.device = 'cpu'
                self.logger.warn('No available GPUs, running on CPU.')
            # TODO: build 3D CNN model with weights and input transforms
            weights = getattr(video_models, self.weight_name).DEFAULT
            self.transforms = weights.transforms()
            base_model = getattr(video_models, self.model_name)(weights=weights)

            self.model = create_feature_extractor(base_model, {self.node_name: 'feature'})
            self.model = self.model.to(self.device).eval()

    def extract_cnn3d_features(self, clip: np.ndarray) -> np.ndarray:
        """
        frame: [T x H x W x C] in uint8 [0, 255]

        Return: Feature, [D]
        """
        # TODO: extract 3D CNN feature for given clip
        # First convert batch into [B x T x C x H x W] with B=1.
        # t = np.random.random((3,4,5,6))
        # t = torch.Tensor(clip)
        # t = t.permute(0,3,1,2)
        # t_clip = clip.transpose(0,3,1,2)
        # b_clip = np.expand_dims(t_clip,0)

        clip = torch.Tensor(clip)
        b_clip = clip.permute(0,3,1,2)
        b_clip.unsqueeze_(dim = 0)
        
        # Then apply self.transforms to batch to get model input.
        tran_clip = self.transforms(b_clip)
        # Finally apply self.model on the input to get features.
        # Wrap the model with torch.no_grad() to avoid OOM.
        with torch.no_grad():
            input = tran_clip.to(self.device)
            output = self.model(input)
        return output["feature"].cpu().numpy()

    def process(self, task):
        task.start(self)
        frames = task.content
        features = self.extract_cnn3d_features(frames)
        task.meta['sequence_id'] = task.meta['batch_id']
        return task.finish(features)
