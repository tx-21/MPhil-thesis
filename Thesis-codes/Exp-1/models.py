import torch.nn as nn
import torch, math
from icecream import ic
import time
from Sklearn_PyTorch.binary_tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor
from Sklearn_PyTorch.utils import sample_vectors, sample_dimensions

class TorchRandomForestRegressor(torch.nn.Module):
    """
    Torch random forest object used to solve regression problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The random forest is based on
    :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` which are built during the :func:`fit` and called
    recursively during the :func:`predict`.

    Args:
        nb_trees (:class:`int`): Number of :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` used to fit the
            classification problem.
        nb_samples (:class:`int`): Number of vector samples used to fit each
            :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor`.
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.
        bootstrap (:class:`bool`): If set to true, a sample of the dimensions of the input vectors are made during the
            fitting and the prediction.

    """
    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, values):
        """
        Function which must be used after the initialisation to fit the random forest and build the successive
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to solve a specific classification problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions_vectors).
            values(:class:`torch.FloatTensor`): Values tensor used to fit the decision tree. It represents the values
                associated to each vectors and must correspond to the following shape (num_vectors,
                num_dimensions_values).

        """
        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeRegressor(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, values, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, values)
            self.trees.append(tree)

    def predict(self, vector):
        """
        Function which must be used after the the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the random forest.

        """
        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions_sum += tree.predict(sampled_vector)
        output = predictions_sum/len(self.trees)
        return output

class model_MLP_1(torch.nn.Module):
    def __init__(self, n_input=1, n_hidden=10, n_batch=1, n_output=1):
        super(model_MLP_1, self).__init__()
        self.input_size = n_input
        self.hidden_size  = n_hidden
        self.batch_size = n_batch
        self.output_size = n_output
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, src, device):
        output = self.fc1(src[:,:,0].unsqueeze(-1))
        output = self.relu(output)
        output = self.fc2(output)
        return output  

class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1 = nn.Conv1d(1,3, kernel_size = 2, stride = 1, padding =1)
        self.bn1 = nn.BatchNorm1d(3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(3, 5, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(5)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(5, 5, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(5)
        self.dropout3 = nn.Dropout(0.3)
    
    self.fc = nn.Linear(5, num_classes)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

class RNN_Model(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons, n_outputs):
        super(RNN_Model, self).__init__()
        self.batch_size=batch_size
        self.n_inputs=n_inputs
        self.n_neurons=n_neurons
        self.n_outputs=n_outputs
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.zeros(batch_size, n_neurons) # initialize hidden state
        self.FC = nn.Linear(n_neurons, n_outputs)
    def forward(self, X):
        self.hx = torch.zeros(self.batch_size, self.n_neurons)
        self.hx = self.rnn(X, self.hx)
        out = self.FC(self.hx)  
        out=out.reshape(out.size(1))
        return out

class model_LSTM_1(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(model_LSTM_1, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 1, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        h_t, c_t = self.lstm(src[:,:,0].unsqueeze(-1), (h_t,c_t))
        output = self.linear(h_t)
        return output

class model_MLP_7(torch.nn.Module):
    def __init__(self, n_input=7, n_hidden=10, n_batch=1, n_output=1):
        super(model_MLP_7, self).__init__()
        self.input_size = n_input
        self.hidden_size  = n_hidden
        self.batch_size = n_batch
        self.output_size = n_output
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
            
    def forward(self, src, device):
        output = self.fc1(src)
        output = self.relu(output)
        output = self.fc2(output)
        return output  

class model_LSTM_7(nn.Module):
    def __init__(self, n_hidden=21): #was 21 originally
        super(model_LSTM_7, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = 7, hidden_size = self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1) #predict 1-step

    def forward(self, src, device):
        h_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        c_t = torch.zeros(self.n_layers,1,self.n_hidden).double()
        x = src
        h_t, c_t = self.lstm(x, (h_t,c_t))
        output = self.linear(h_t)
        return output

class Transformer(nn.Module):
    # d_model : number of features
    # only doing self-att, mask is required
    def __init__(self,feature_size=7,num_layers=1,dropout=0): # changed from 3 to 1 #stacking the decoder blocks three times
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device): 
        x = src
        mask = self._generate_square_subsequent_mask(len(x)).to(device)
        output = self.transformer_encoder(x,mask)
        output = self.decoder(output)
        return output

