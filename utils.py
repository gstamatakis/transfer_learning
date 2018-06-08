from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import os.path
import random
import re

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from parser_options import FLAGS


# From Google
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      An OrderedDict containing an entry for each label subfolder, with images
      split into training, testing, and validation sets within each label.
      The order of items defines the class indices.
    """
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < FLAGS.min_images_per_label:
            tf.logging.warning('WARNING: Folder has less than {0} images, which may cause issues.'.format(FLAGS.min_images_per_label))
        elif len(file_list) > FLAGS.max_images_per_label:
            tf.logging.warning('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, FLAGS.max_images_per_label))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (FLAGS.max_images_per_label + 1)) * (100.0 / FLAGS.max_images_per_label))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index.

    Args:
      image_lists: Training images for each label.
      label_name: The label name.
      index: Int offset of the image we want. This will be moded (%) by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training images.
      category: Name string of set to pull images from - training, testing, or validation.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def create_module_graph(module_spec):
    """Creates a graph and loads Hub Module into it.

    Args:
      module_spec: the hub.ModuleSpec for the image module being used.

    Returns:
      graph: the tf.Graph that was created.
      bottleneck_tensor: the bottleneck values output by the module.
      resized_input_tensor: the input images, resized as expected by the module.
    """
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])  # Setting channels to 3.
        bottleneck_tensor = hub.Module(module_spec)(resized_input_tensor)
    return graph, bottleneck_tensor, resized_input_tensor


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    try:
        # Runs inference on an image to extract the 'bottleneck' summary layer. First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


# From Google
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of which set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The output tensor for the bottleneck values.
      module_name: The name of the image module being used.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)

    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    module_name = (module_name.replace('://', '~')  # URL scheme.
                   .replace('/', '~')  # URL and Unix paths.
                   .replace(':', '~').replace('\\', '~'))  # Windows paths.

    bottleneck_path = get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + module_name + '.txt'

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    bottleneck_values = None
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_cached_bottlenecks(sess, images, how_many, category, bottleneck_dir, image_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      images: OrderedDict of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
      module_name: The name of the image module being used.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(images.keys())
    bottlenecks = []
    labels = []
    filenames = []
    if how_many >= 0:
        for _ in range(how_many):  # Retrieve a random sample of bottlenecks.
            label_index = random.randrange(class_count)
            label_name = list(images.keys())[label_index]
            image_idx = random.randrange(FLAGS.max_images_per_label + 1)
            image_name = get_image_path(images, label_name, image_idx, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, images, label_name, image_idx, image_dir, category, bottleneck_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            labels.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(images.keys()):
            for image_idx, image_name in enumerate(images[label_name][category]):
                image_name = get_image_path(images, label_name, image_idx, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, images, label_name, image_idx, image_dir, category, bottleneck_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                labels.append(label_index)
                filenames.append(image_name)
    return bottlenecks, labels, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

    Args:
      sess: Current TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      image_dir: Root folder string of the subfolder containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    labels = []
    for _ in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(FLAGS.max_images_per_label + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        labels.append(label_index)
    return bottlenecks, labels


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, module_spec):
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.convert_image_dtype(tf.image.decode_jpeg(jpeg_data, channels=input_depth), tf.float32)
    scale_value = tf.multiply(tf.constant(1.0 + (random_crop / 100.0)), tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + (random_scale / 100.0)))
    precrop_shape = tf.cast(tf.stack([tf.multiply(scale_value, input_height), tf.multiply(scale_value, input_width)]), dtype=tf.int32)
    precropped_image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(decoded_image, 0), precrop_shape), axis=[0])
    cropped_image = tf.random_crop(precropped_image, [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image  # Do nothing
    brightness_value = tf.random_uniform(shape=[], minval=1.0 - (random_brightness / 100.0), maxval=1.0 + (random_brightness / 100.0))
    distort_result = tf.expand_dims(tf.multiply(flipped_image, brightness_value), 0, name='DistortResult')
    return jpeg_data, distort_result


def add_retrain_operations(class_count, final_tensor_name, bottleneck_tensor, is_training):
    """Adds a new softmax and fully-connected layer for training and eval."""

    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        label_input = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

    initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
    layer_weights = tf.Variable(initial_value, name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # If this is an eval graph, we dont need to add loss ops or optimizer.
    if not is_training:
        return None, None, bottleneck_input, label_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=label_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # Chose an optimiser
    with tf.name_scope('train'):
        opt = FLAGS.optimizer
        if opt == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        elif opt == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta1, epsilon=FLAGS.epsilon)
        elif opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, decay=FLAGS.decay, momentum=FLAGS.momentum, epsilon=FLAGS.epsilon)
        else:
            raise RuntimeError('Unknown optimizer: {0}'.format(opt))

        train_step = optimizer.minimize(cross_entropy_mean)
    return train_step, cross_entropy_mean, bottleneck_input, label_input, final_tensor


def add_evaluation_step(result_tensor, label_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      label_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, label_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def build_eval_session(module_spec, class_count):
    """Builds an restored eval session without train operations for exporting."""

    eval_graph, bottleneck_tensor, resized_input_tensor = create_module_graph(module_spec)

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, bottleneck_input, label_input, final_tensor) = add_retrain_operations(class_count, FLAGS.final_tensor_name, bottleneck_tensor, is_training=False)
        # Now we need to restore the values from the training graph to the eval graph.
        tf.train.Saver().restore(eval_sess, FLAGS.checkpoint_path)
        evaluation_step, prediction = add_evaluation_step(final_tensor, label_input)

    return eval_sess, resized_input_tensor, bottleneck_input, label_input, evaluation_step, prediction


def save_graph_to_file(graph_file_name, module_spec, class_count):
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return jpeg_data, resized_image


def export_model(module_spec, class_count, saved_model_dir):
    """Exports model for serving.

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: The number of classes.
      saved_model_dir: Directory in which to save exported model and variables.
    """
    # The SavedModel should hold the eval graph.
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph
    with graph.as_default():
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name('final_result:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                             legacy_init_op=legacy_init_op)
        builder.save()
