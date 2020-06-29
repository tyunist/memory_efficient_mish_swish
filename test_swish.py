'''
test_swish.py 
Test Swish() implementation

# Based on: Ceshine Lee 
https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76

# Author: Ty Nguyen
# Contact: tynguyen.tech@gmail.com
'''

import gc
import torch
import torch.nn as nn
import numpy as np
from swish import Swish 

from sklearn.datasets import make_classification

def get_model(swish_module):
    # Deliberately make the model very large
    width = 2 ** 19
    return nn.Sequential(
        nn.Linear(256, width),
        swish_module(inplace=True),
        nn.BatchNorm1d(width),
        nn.Linear(width, 1)
    )


def print_parameter_count(model):
    print("# of parameters: {:,d}".format(
        np.sum(list(p.numel() for p in model.parameters()))))
    print("# of trainable parameters: {:,d}".format(
        np.sum(list(p.numel() for p in model.parameters() if p.requires_grad))))


class PlainSwish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)

#===============================================================================
def test_swish_forward_backward():
    torch.manual_seed(0)
    import numpy as np 
    np.random.seed(0)

    total_test = 20
    n_correct  = 0
    for t in range(total_test):
        np_input    = np.random.randn(20, 30)
        plain_input = torch.tensor([np_input], requires_grad=True)
        eff_input   = torch.tensor([np_input], requires_grad=True)

        plain_swish = PlainSwish()
        eff_swish   = Swish()

        plain_output = plain_swish(plain_input).sum()
        eff_output   =  eff_swish(eff_input).sum()

        print("=======================\nForward:")
        assert plain_output - eff_output < 1e-9, "Outpus must match!"

        print("=======================\nBackward:")
        eff_output.backward()
        eff_grad    = eff_input.grad 
        #print("Eff grad:\n", eff_grad)

        print("----------------")
        plain_output.backward()
        plain_grad  = plain_input.grad 
        
        #print("Plain grad:\n", plain_grad)

        #print("----------------")
        delta_grad = torch.abs(plain_grad - eff_grad).max()
        print("Max delta grad:\n", delta_grad)

        assert delta_grad < 1e-15, "Grads must match!" 
        print("==========================\n")
        print("Successful!")
        n_correct += 1
        
    print("==========================\n")
    print("Completed! Succeeded %d/%d!"%(n_correct, total_test))


def test_swish_memory():
    X, y = make_classification(
        n_samples=1024, 
        n_features=256, 
        n_informative=128, 
        n_redundant=0, 
        n_repeated=0, 
        n_classes=2, 
        n_clusters_per_class=2, 
        flip_y=0.01, 
        class_sep=1.0, 
        hypercube=True, 
        shuffle=True, 
        random_state=42
    )


    criterion = nn.BCEWithLogitsLoss()
    batch_size = 128

    model = get_model(PlainSwish).cuda()
    print_parameter_count(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    print("begin:", torch.cuda.memory_allocated() / 1024 ** 2)


    for i in range(0, 1024, batch_size):

        Xt, yt = torch.tensor(X[i:i+batch_size], dtype=torch.float).cuda(), torch.tensor(y[i:i+batch_size], dtype=torch.float).cuda()
        print("data:", torch.cuda.memory_allocated() / 1024 ** 2)
        pred = model(Xt)[:, 0]
        print("forw:", torch.cuda.memory_allocated() / 1024 ** 2)
        loss = criterion(pred, yt)
        # print(loss)
        print("loss:", torch.cuda.memory_allocated() / 1024 ** 2)
        loss.backward()
        print("back:", torch.cuda.memory_allocated() / 1024 ** 2)
        optimizer.step()
        optimizer.zero_grad()
        print("step:", torch.cuda.memory_allocated() / 1024 ** 2)
        print("=" * 20)


    del optimizer, model, Xt, yt, loss, pred
    gc.collect()
    print("end:", torch.cuda.memory_allocated() / 1024 ** 2)


    print("===============================================")
    print("Custom  Swish")
    model = get_model(Swish).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    torch.cuda.memory_allocated() / 1024


    for i in range(0, 1024, batch_size):
        Xt, yt = torch.tensor(X[i:i+batch_size], dtype=torch.float).cuda(), torch.tensor(y[i:i+batch_size], dtype=torch.float).cuda()
        print("data:", torch.cuda.memory_allocated() / 1024 ** 2)
        pred = model(Xt)[:, 0]
        print("forw:", torch.cuda.memory_allocated() / 1024 ** 2)
        loss = criterion(pred, yt)
        # print(loss)
        print("loss:", torch.cuda.memory_allocated() / 1024 ** 2)
        loss.backward()
        print("back:", torch.cuda.memory_allocated() / 1024 ** 2)
        optimizer.step()
        optimizer.zero_grad()
        print("step:", torch.cuda.memory_allocated() / 1024 ** 2)
        print("=" * 20)


    del optimizer, model, Xt, yt, loss, pred
    gc.collect()
    print("end:", torch.cuda.memory_allocated() / 1024 ** 2)

if __name__=="__main__":
    test_swish_memory()
    test_swish_forward_backward()
