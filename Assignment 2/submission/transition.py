class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
	if not conf.buffer or not conf.stack:
	    return -1

	idx_wi = conf.stack[-1]

	# The token should not be the artificial root node 0
	if idx_wi == 0:
	    return -1
	# The token should not have a head
	has_head = False
	for (k, l, i) in conf.arcs:
            if i == idx_wi:
                has_head = True
                break
	if has_head:
            return -1
	
	idx_wj = conf.buffer[0]
	# remove from the stack
        del conf.stack[-1]
	# add to arc list
        conf.arcs.append((idx_wj, relation, idx_wi))

        #raise NotImplementedError('Please implement left_arc!')
        #return -1

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        # pop one item from the buffer
        idx_wj = conf.buffer.pop(0)
        # add to the stack
        conf.stack.append(idx_wj)
        # add to the arc list
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
	    return -1
        # the word should have a head
	has_head = False
	for (k, l, i) in conf.arcs:
	    if i == conf.stack[-1]:
	        has_head = True
	if not has_head:
	    return -1
	# remove from the stack
	del conf.stack[-1]
	return conf
	#raise NotImplementedError('Please implement reduce!')
        #return -1

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer:
	    return -1
        # pop the item from the buffer and add it to the stack
	idx_wj = conf.buffer.pop(0)
	conf.stack.append(idx_wj)
	#raise NotImplementedError('Please implement shift!')
        #return -1
