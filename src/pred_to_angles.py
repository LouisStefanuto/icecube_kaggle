import torch

def from_angles(out, labels):
    az_true, zen_true = labels[:,0], labels[:,1]
    az_pred, zen_pred = out[:,0], out[:,1]
    
    return az_true, zen_true, az_pred, zen_pred


def from_cossin(out, labels):
    az_cos_pred, az_sin_pred, zen_cos_pred, zen_sin_pred = [out[:,i] for i in range(4)]
    az_cos_true, az_sin_true, zen_cos_true, zen_sin_true = [labels[:,i] for i in range(4)]

    az_pred, zen_pred = torch.atan2(az_sin_pred, az_cos_pred), torch.atan2(zen_sin_pred, zen_cos_pred)
    az_true, zen_true = torch.atan2(az_sin_true, az_cos_true), torch.atan2(zen_sin_true, zen_cos_true)
    
    return az_true, zen_true, az_pred, zen_pred


def from_xyz(out, labels):
    x_pred, y_pred, z_pred = [out[:,i] for i in range(3)]
    x_true, y_true, z_true = [labels[:,i] for i in range(3)]

    az_pred, zen_pred = torch.atan2(y_pred, x_pred), torch.arccos(z_pred / torch.sqrt(x_pred**2 + y_pred**2 + z_pred**2))
    az_true, zen_true = torch.atan2(y_true, x_true), torch.arccos(z_true / torch.sqrt(x_true**2 + y_true**2 + z_true**2))  
    
    return az_true, zen_true, az_pred, zen_pred