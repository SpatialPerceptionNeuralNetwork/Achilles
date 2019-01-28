import tensorflow as _tf
import dataset.csv as csv
import os as _os
from utils import logutil as _logutil
from . import augmentation as _augmentation

_tde = _tf.data.experimental
_logging = _logutil.get_logger()
_FLAGS = _tf.app.flags.FLAGS


def load(machine=None):
    csv.inspect()
    if machine is not None:
        machine.dataset_loader = Loader()
    else:
        return Loader()


class Loader():
    def __init__(self):
        self._streams = {}
        self._generate_stream()
        pass

    def _generate_stream(self):
        for key in [key for key in _FLAGS if str(key).endswith('csv')]:
            csv_file = _FLAGS[key]._value
            csv_path = _os.path.join(_FLAGS.dataset_dir, _FLAGS.type,
                                     csv_file)
            key = str(key).split('_')[0]

            for suffix, phase in [("train", _FLAGS.phase_train), ("eval", _FLAGS.phase_eval),
                                  ("sampling", _FLAGS.phase_sampling)]:
                data_pool = _tf.data.TextLineDataset([csv_path])
                augs = []
                if key.find('train') >= 0:
                    batch_size = _FLAGS.trainset_batch_size
                    for _key in [_key for _key in _FLAGS]:
                        if str(_key).startswith('augment'):
                            aug_flag = _FLAGS[_key]._value
                            if aug_flag:
                                augs.append(str(_key).split('_')[1])
                elif key.find('valid') >= 0:
                    batch_size =3
                elif key.find('test') >= 0:
                    batch_size = 3
                elif key.find('ipiu') >= 0:
                    batch_size = 3
                else:
                    raise AttributeError("Invalid batch_size")

                data_pool = data_pool.apply(_tde.shuffle_and_repeat(_FLAGS.buffer_size))
                # data_pool = data_pool.apply(_tde.shuffle_and_repeat(1))
                data_pool = data_pool.apply(_tde.map_and_batch(
                    map_func=lambda row: self._read_row(row, phase, augs),
                    batch_size=batch_size,
                    num_parallel_batches=_FLAGS.num_parallel_batches
                ))
                batch_stream = data_pool.make_one_shot_iterator().get_next()

                print("Stream generated... " + key + '_phase_' + suffix)
                self._streams[key + '_phase_' + suffix] = batch_stream

    def _read_row(self, csv_row, phase, augs):
        def _pre_processing(stream, phase):
            def _random_crop(stream):
                x, y = stream
                assert x.shape == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w, 3)
                assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
                size_h, size_w = (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
                off_w = _tf.cast(_tf.random_uniform([], 0, size_w - _FLAGS.dim_input_w), _tf.int32)
                off_h = _tf.cast(_tf.random_uniform([], 0, size_h - _FLAGS.dim_input_h), _tf.int32)

                try:
                    _x = _tf.image.crop_to_bounding_box(x, offset_height=off_h, offset_width=off_w,
                                                        target_width=_FLAGS.dim_input_w,
                                                        target_height=_FLAGS.dim_input_h)
                    _y = _tf.image.crop_to_bounding_box(y, offset_height=off_h, offset_width=off_w,
                                                        target_width=_FLAGS.dim_input_w,
                                                        target_height=_FLAGS.dim_input_h)
                    x = _x
                    y = _y
                except Exception as e:
                    print(e)

                return x, y

            # def _center_crop(stream):
            #     x, y = stream
            #     assert x.shape == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w, 3)
            #     assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
            #     size_h, size_w = (_FLAGS.dim_input_h, _FLAGS.dim_input_w)
            #
            #     try:
            #         _x = _tf.image.resize_image_with_crop_or_pad(x, size_h, size_w)
            #         _y = _tf.image.resize_image_with_crop_or_pad(y, size_h, size_w)
            #
            #         x = _x
            #         y = _y
            #     except Exception as e:
            #         print(e)
            #
            #     return x, y

            if phase == _FLAGS.phase_train:
                if str(_FLAGS.type).startswith("KITTI"):
                    return _random_crop(stream)
                else:
                    return stream
            elif phase == _FLAGS.phase_eval or phase == _FLAGS.phase_sampling or phase==_FLAGS.phase_ipiu:
                return stream
            else:
                raise AttributeError(phase)

        rgb_name, gt_name = _tf.decode_csv(csv_row, record_defaults=[[""], [""]])
        rgb_img = self._load_img(rgb_name, channels=3)
        gt_img = self._load_img(gt_name)

        (rgb_img, gt_img) = _augmentation.augment((rgb_img, gt_img), augs)
        (rgb_img, gt_img) = _pre_processing((rgb_img, gt_img), phase)

        return rgb_img, gt_img

    def _load_img(self, filename, channels=1):
        file = _tf.read_file(filename)
        if channels == 3:
            image = _tf.image.decode_jpeg(file, channels)
            image = _tf.cast(image, _tf.float32)
        elif channels == 1:

            if str(_FLAGS.type).startswith("KITTI"):
                image = _tf.image.decode_png(file, channels, dtype=_tf.uint16)
                image = _tf.cast(image, _tf.float32)
                image = image / 256.0
                image = _tf.where(_tf.less_equal(image, 0.0), -_tf.ones_like(image), image)
                image = _tf.where(_tf.greater_equal(image, _FLAGS.GT_maxima), _tf.ones_like(image) * _FLAGS.GT_maxima,
                                  image)
            elif str(_FLAGS.type).startswith("NYU"):
                image = _tf.image.decode_png(file, channels)
                image = _tf.cast(image, _tf.float32)
                image = image / 256.0 * _FLAGS.GT_maxima
            else:
                image = _tf.image.decode_png(file, channels)
                image = _tf.cast(image, _tf.float32)
        else:
            return
        image = _tf.image.resize_images(image,
                                        (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w))
        return image

    def get_stream_names(self):
        return list(self._streams.keys())

    def get_stream_batch(self, sess, phase, stream_name='trainset'):
        try:
            if phase == _FLAGS.phase_train:
                stream_name += '_phase_train'
            elif phase == _FLAGS.phase_eval:
                stream_name += '_phase_eval'
            elif phase == _FLAGS.phase_sampling:
                stream_name += '_phase_sampling'

            stream = self._streams[stream_name]
        except KeyError:
            if sess is not None:
                sess.close()
            raise _tf.errors.InternalError(None, None, message="Stream named '%s' doesn't exists" % stream_name)

        if sess is not None:
            for retries in range(1, 6):
                try:
                    input, label = sess.run(stream)
                except KeyboardInterrupt:
                    sess.close()
                    raise _tf.errors.CancelledError(None, None, message="KeyboardInterrupt")
                except:
                    _logging.warning('\nGet trainset on batch', 'retry...({})'.format(retries))
                    continue
                else:
                    return input, label
            raise _tf.errors.DataLossError(None, None, message='Retries expired')
        else:
            return stream

        pass
