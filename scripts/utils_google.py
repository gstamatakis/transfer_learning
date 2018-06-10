import collections
import hashlib
import os.path
import random
import re

import numpy as np

from scripts.utils import *


# All functions in this file have been taken from the tensorflow slim main github repo and were slightly altered.

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


# From Google tensorflow slim repo
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


# From Google tensorflow slim repo
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


# From Google tensorflow slim repo
def get_cached_bottlenecks(sess, images, how_many, category, bottleneck_dir, image_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
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
    if how_many >= 0:  # Retrieve a random sample of bottlenecks.
        for _ in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(images.keys())[label_index]
            image_idx = random.randrange(FLAGS.max_images_per_label + 1)
            image_name = get_image_path(images, label_name, image_idx, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, images, label_name, image_idx, image_dir, category, bottleneck_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            labels.append(label_index)
            filenames.append(image_name)
    else:  # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(images.keys()):
            for image_idx, image_name in enumerate(images[label_name][category]):
                image_name = get_image_path(images, label_name, image_idx, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, images, label_name, image_idx, image_dir, category, bottleneck_dir, jpeg_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                labels.append(label_index)
                filenames.append(image_name)
    return bottlenecks, labels, filenames


# Altered from the google tf-slim repo.
def get_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

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
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: tf.gfile.FastGFile(image_path, 'rb').read()})
        bottlenecks.append(np.squeeze(sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})))
        labels.append(label_index)
    return bottlenecks, labels
