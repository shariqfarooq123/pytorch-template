import torch
from settings import SAVE_DIR
import glob
import os

class ResourceNotFoundException(Exception):
    pass

def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    return load_state_dict(model, ckpt)


def load_state_dict_from_url(model, url, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, **kwargs)
    return load_state_dict(model, state_dict)

def pattern_to_local_resource(pattern, local_dir=SAVE_DIR):
    """ Find the local resource matching the pattern under local_dir
    """
    files = glob.glob(os.path.join(local_dir, f'*{pattern}*'))
    if len(files) == 0:
        raise ResourceNotFoundException(f"Could not find any local resource matching {pattern}")
    elif len(files) > 1:
        raise ValueError(f"Found multiple local resources matching {pattern}")
    return files[0]


def fetch_or_load_pattern_from_remote(remote_host_name, remote_root_dir, pattern, local_dir=SAVE_DIR):
    """ Fetches a file from a remote host if it does not exist locally
    """
    try:
        local_resource = pattern_to_local_resource(pattern, local_dir)
    except ResourceNotFoundException:
        from utils.remote_utils import Remote
        with Remote(remote_host_name) as remote:
            found = remote.get_by_pattern(remote_root_dir, pattern, local_dir)
            if not found:
                raise ResourceNotFoundException(f"Could not find any resource matching {pattern} on {remote_host_name}")
            local_resource = pattern_to_local_resource(pattern, local_dir)
    return local_resource

def fetch_or_load_pattern_from_any_remote(remote_root_dir, pattern, local_dir=SAVE_DIR, remote_host_names=None):
    """ Fetches a file from any of the remote hosts if it does not exist locally
    """
    if remote_host_names is None:
        from utils.remote_utils import REMOTE_HOSTS
        remote_host_names = REMOTE_HOSTS

    for remote_host_name in remote_host_names:
        try:
            return fetch_or_load_pattern_from_remote(remote_host_name, remote_root_dir, pattern, local_dir)
        except ResourceNotFoundException:
            continue
    raise ResourceNotFoundException(f"Could not find any resource matching {pattern} on any of the remote hosts {remote_host_names}")

def load_state_from_resource(model, resource):
    """Loads weights to the model from a given resource. A resource can be of following types:

        1. URL. Prefixed with "url::"
                e.g. url::http(s)://url.resource.com/ckpt.pt

        2. Local path. Prefixed with "local::"
                e.g. local::/path/to/ckpt.pt
        3. Remote path. Prefixed with "remote::"
                e.g. remote::host_name::/path/to/ckpt_dir::ckpt.pt


    Args:
        model (torch.nn.Module): Model
        resource (str): resource string

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    print(f"Using pretrained resource {resource}")

    if resource.startswith('url::'):
        url = resource.split('url::')[1]
        return load_state_dict_from_url(model, url, progress=True)
    
    elif resource.startswith('local::pattern::'):
        pattern = resource.split('local::pattern::')[1]
        return load_wts(model, pattern_to_local_resource(pattern))

    elif resource.startswith('local::'):
        path = resource.split('local::')[1]
        return load_wts(model, path)
    
    elif resource.startswith('anyremote::'):
        remote_root_dir, pattern = resource.split('::')[1:]
        local_resource = fetch_or_load_pattern_from_any_remote(remote_root_dir, pattern)
        return load_wts(model, local_resource)
    
    elif resource.startswith('remote::'):
        remote_host_name, remote_root_dir, pattern = resource.split('::')[1:]
        local_resource = fetch_or_load_pattern_from_remote(remote_host_name, remote_root_dir, pattern)
        return load_wts(model, local_resource)
        
    else:
        raise ValueError("Invalid resource type, only url::, local::, local::pattern::, anyremote::, remote::, are supported")
