import tensorflow as _tf

_flags = _tf.app.flags
_FLAGS = _flags.FLAGS

# ===================
# COMMON
# ===================
_flags.DEFINE_string("about", "Created by Sangwon Kim", "[COMMON] about")
_flags.DEFINE_string("email1", "swkim@stu.kmu.ac.kr", "[COMMON] email1")
_flags.DEFINE_string("email2", "swkim@eng.ucsd.edu", "[COMMON] email2")

# ===================
# DATASET
# ===================
_flags.DEFINE_string("ext_rgb", "jpg", "[DATASET] Extension of input images")
_flags.DEFINE_string("ext_gt", "png", "[DATASET] Extension of ground-truth images")

_flags.DEFINE_float("INPUT_minima", 0.0, "[DATASET] maximum range in meters")
_flags.DEFINE_float("INPUT_maxima", 255.0, "[DATASET] maximum range in meters")

# _flags.DEFINE_string("type", "KITTI_eigen_dense", "[DATASET] type?")
_flags.DEFINE_string("type", "NYU", "[DATASET] type?")
if str(_FLAGS.type).startswith('KITTI'):
    _flags.DEFINE_string("KITTI_side", "image_02", "[DATASET] which side?")
    _flags.DEFINE_float("GT_minima", 0.0, "[DATASET] maximum range in meters")
    _flags.DEFINE_float("GT_maxima", 80.0, "[DATASET] maximum range in meters")
elif str(_FLAGS.type).startswith('NYU'):
    _flags.DEFINE_float("GT_minima", 0.0, "[DATASET] maximum range in meters")
    _flags.DEFINE_float("GT_maxima", 10.0, "[DATASET] maximum range in meters")
else:
    _flags.DEFINE_float("GT_minima", 0.0, "[DATASET] maximum range in meters")
    _flags.DEFINE_float("GT_maxima", 255.0, "[DATASET] maximum range in meters")

_flags.DEFINE_string("dataset_dir", "../../DATASET", "[DATASET] base dir")
_flags.DEFINE_string("trainset_img_dir", "train-set", "[DATASET] trainset image dir")
_flags.DEFINE_string("validset_img_dir", "valid-set", "[DATASET] validset image dir")
_flags.DEFINE_string("testset_img_dir", "test-set", "[DATASET] testset image dir")
_flags.DEFINE_string("ipiu_img_dir", "ipiu", "[DATASET] testset image dir")
_flags.DEFINE_string("trainset_csv", "train.csv", "[DATASET] trainset csv")
_flags.DEFINE_string("validset_csv", "valid.csv", "[DATASET] validset csv")
_flags.DEFINE_string("testset_csv", "test.csv", "[DATASET] testset csv")
# _flags.DEFINE_string("ipiu_csv", "ipiu.csv", "[DATASET] testset csv")

_flags.DEFINE_string("trainset_stream", "trainset", "[DATASET] trainset csv")
_flags.DEFINE_string("validset_stream", "validset", "[DATASET] validset csv")
_flags.DEFINE_string("testset_stream", "testset", "[DATASET] testset csv")
_flags.DEFINE_string("ipiu_stream", "ipiu", "[DATASET] testset csv")

_flags.DEFINE_integer("buffer_size", 1500, "[DATASET] buffer size")
_flags.DEFINE_integer("num_parallel_batches", 1, "[DATASET] num_parallel_batches")

_flags.DEFINE_boolean("augment_scale", False, "[DATASET] augmentation")
_flags.DEFINE_boolean("augment_trans", False, "[DATASET] augmentation")
_flags.DEFINE_boolean("augment_color", False, "[DATASET] augmentation")
_flags.DEFINE_boolean("augment_flips", False, "[DATASET] augmentation")
_flags.DEFINE_boolean("augment_rot", False, "[DATASET] augmentation")
_flags.DEFINE_boolean("augment_random", True, "[DATASET] augmentation")

# ===================
# SAMPLING
# ===================
_flags.DEFINE_string("sampling_dir", "./_sampling", "[SAMPLING] sampling dir")
_flags.DEFINE_string("learn_sampling_dir", "trainset", "[SAMPLING] sampling dir")
_flags.DEFINE_string("valid_sampling_dir", "validset", "[SAMPLING] sampling dir")
_flags.DEFINE_string("test_sampling_dir", "testset", "[SAMPLING] sampling dir")

# ===================
# META
# ===================
_flags.DEFINE_string("meta_dir", "./_meta", "[META] meta dir")
_flags.DEFINE_boolean("restore", True, "[META] restore")
_flags.DEFINE_boolean("backup", True, "[META] backup")
_flags.DEFINE_string("feature_extractor", "resnet_v1_101", "[META] feature_extractor")
_flags.DEFINE_boolean("use_pretrained_data", True, "[META] backup")

_flags.DEFINE_string("pretrained_meta_dir", "./pretrained_meta", "[META] meta dir")

_flags.DEFINE_string("log_dir", "./_log", "[META] log dir")

# ===================
# HYPER-P
# ===================
_flags.DEFINE_integer("dim_dataset_h", 385, "")
if str(_FLAGS.type).startswith('NYU'):
    _flags.DEFINE_integer("dim_dataset_w", 513, "")
    _flags.DEFINE_integer("num_frag", 1, "")
elif str(_FLAGS.type).startswith('KITTI'):
    _flags.DEFINE_integer("dim_dataset_w", 1242, "")
    _flags.DEFINE_integer("num_frag", 4, "")
elif str(_FLAGS.type).startswith('Make3D'):
    _flags.DEFINE_integer("dim_dataset_w", 1242, "")
else:
    raise ValueError("_FLAGS.type:(%s) doesn't exist" % _FLAGS.type)

_flags.DEFINE_integer("dim_input_h", 385, "")
_flags.DEFINE_integer("dim_input_w", 513, "")
_flags.DEFINE_integer("dim_output_h", 385, "")
_flags.DEFINE_integer("dim_output_w", 513, "")

_flags.DEFINE_integer("default_batch_size", 3, "")
_flags.DEFINE_integer("trainset_batch_size", 3, "")
if str(_FLAGS.type).startswith('KITTI_eigen'):
    _flags.DEFINE_integer("testset_batch_size", 697, "")
    _flags.DEFINE_integer("validset_batch_size", 888, "")
    _flags.DEFINE_integer("ipiu_batch_size", 6, "")
else:
    _flags.DEFINE_integer("testset_batch_size", 3, "")
    _flags.DEFINE_integer("validset_batch_size", 3, "")

_flags.DEFINE_integer("num_classes", 142, "")
_flags.DEFINE_integer("num_bins", int(_FLAGS.num_classes / 2), "")
_flags.DEFINE_float("dsc_shift", 1.0, "")
_flags.DEFINE_integer("output_feature_stride", 8, "")

_flags.DEFINE_float("learning_rate", 0.0001, "")

_flags.DEFINE_integer("max_number_of_steps", 30000000, "")
_flags.DEFINE_integer("backup_every_n_steps", 200, "")

_flags.DEFINE_boolean("use_tensorboard", True, "[META] backup")
_flags.DEFINE_integer("tensorboard_interval", 50, "[META] backup")
_flags.DEFINE_boolean("use_sampling", True, "[META] backup")
_flags.DEFINE_integer("sampling_interval", 1000, "[META] backup")

_flags.DEFINE_integer("phase_train", 0, "")
_flags.DEFINE_integer("phase_eval", 1, "")
_flags.DEFINE_integer("phase_sampling", 2, "")
_flags.DEFINE_integer("phase_ipiu", 3, "")
