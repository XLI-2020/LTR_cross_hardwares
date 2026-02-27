import torch.nn 

from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from comparisonModels.BAO.TreeConvolution.utils import prepare_trees

# This code snippet was mostly extracted, with some renaming from: 
# https://github.com/learnedsystems/BaoForPostgreSQL/blob/master/bao_server/net.py
class LTRComparisonNet(torch.nn.Module):
    def __init__(self, in_channels_1, in_channels_2):
        super(LTRComparisonNet, self).__init__() 
        self.input_channels_1 = in_channels_2 # a bit confusing but it wouldnt work otherwise
        self.input_channels_2 = in_channels_1
        
        self.tree_conv = torch.nn.Sequential(
            BinaryTreeConv(self.input_channels_2, 256),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1)
        )
        

    #def get_input_channels(self):
    #    return self.input_channels
    
    def forward(self, samples_vec, samples_tree):

        
        trees = prepare_trees(samples_tree, get_features, get_left_child, get_right_child, cuda=False)
        return self.tree_conv(trees)
        
    
    def predict_all(self, samples_vec, samples_tree):
        tree = prepare_trees(samples_tree, get_features, get_left_child, get_right_child)
        return self.tree_conv(tree)

        
                              
