import tensorflow as _tf

_slim = _tf.contrib.slim


def anonymous_1(feats, feats_endpoints, is_training):
    # 49x65x2048 --> 385x513x142
    _activation = _tf.nn.relu
    _branches = []
    with _tf.variable_scope('d_conv_anonymous_1'):

        low_level_feature = feats_endpoints['vgg_16/conv2/conv2_1']
        _branches.append(low_level_feature)

        concat_1 = _tf.concat([feats, feats_endpoints['vgg_16/conv5/conv5_1']], axis=3)
        concat_1 = _slim.conv2d_transpose(concat_1, 512, 3, 2, activation_fn=_activation)

        concat_2 = _tf.concat([concat_1, feats_endpoints['vgg_16/conv4/conv4_1']], axis=3)
        concat_2 = _slim.conv2d_transpose(concat_2, 512, 3, 2, activation_fn=_activation)

        with _tf.variable_scope('aspp_0'):
            aspp_1 = _slim.conv2d(concat_2, 256, 1, rate=1, activation_fn=_activation)
            conv6_2 = _slim.conv2d_transpose(aspp_1, 256, 3, 2, activation_fn=_activation)
            _branches.append(conv6_2)

        for i, rate in enumerate([6, 12, 18], 1):
            with _tf.variable_scope('aspp_%d' % i):
                _aspp = _slim.conv2d(concat_2, 256, 1, rate=rate, activation_fn=_activation)
                _conv = _slim.conv2d_transpose(_aspp, 256, 3, 2, activation_fn=_activation)
                _branches.append(_conv)

        with _tf.variable_scope('post_proc'):
            conv6_concat = _tf.concat(_branches, axis=3)
            conv7 = _slim.dropout(conv6_concat, 0.5, is_training=is_training)

            # TOP: (?,49,65,2048)
            conv8 = _slim.conv2d(conv7, 142, 1, 1, activation_fn=None)
            # TOP: (?,49,65,142)
            conv8 = _tf.image.resize_bilinear(conv8, [385, 513])
        out = conv8

    trainable_variables = _slim.get_trainable_variables(scope='d_conv_anonymous_1')
    for var in _tf.global_variables(scope='d_conv_anonymous_1'):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)
    """[None, 385, 513, 142]"""
    return out, {'d_conv_anonymous_1': trainable_variables}
