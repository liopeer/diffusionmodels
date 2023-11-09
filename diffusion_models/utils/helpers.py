class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def bytes_to_gb(bytes: int):
    kb = bytes / 1024
    mb = kb / 1024
    gb = mb / 1024
    return gb