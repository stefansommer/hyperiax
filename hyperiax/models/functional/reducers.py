

def product_fuse_children(axis=-1):
    """ Fuse across children 

    :param axis:  axis (int, optional): the axis to take product over. Defaults to -1.
    """
    def _fuse(**kwargs):
        """ _fuse

        :return: returns a function that takes product over an axis of keyed values prefixed with `child_`
        """
        return {k[6:]:v.prod(axis=axis) for k,v in kwargs.items() if k.startswith('child_')}
    return _fuse

def sum_fuse_children(axis=-1):
    """ Fuse across children 

    :param axis:  axis (int, optional): the axis to sum over. Defaults to -1.
    """
    def _fuse(**kwargs):
        """ _fuse

        :return: returns a function that sums an axis of keyed values prefixed with `child_`
        """
        return {k[6:]:v.sum(axis=axis) for k,v in kwargs.items() if k.startswith('child_')}
    return _fuse
