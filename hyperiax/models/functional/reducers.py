

def sum_fuse_children(axis=-1):
    """returns a function that sums an axis of keyed values prefixed with `child_`

    Args:
        axis (int, optional): the axis to sum over. Defaults to -1.
    """
    def _fuse(**kwargs):
        return {k[6:]:v.sum(axis=axis) for k,v in kwargs.items() if k.startswith('child_')}
    return _fuse


def product_fuse_children(axis=-1):
    """returns a function that computes the product of an axis of keyed values prefixed with `child_`

    Args:
        axis (int, optional): the axis to compute product over. Defaults to -1.
    """
    def _fuse(**kwargs):
        return {k[6:]:v.prod(axis=axis) for k,v in kwargs.items() if k.startswith('child_')}
    return _fuse

