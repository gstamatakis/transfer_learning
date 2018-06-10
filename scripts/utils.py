import os.path

import tensorflow as tf
import tensorflow_hub as hub

from scripts.parser_options import FLAGS


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns the full path of an image from a given image list."""
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def create_module_graph(module_spec):
    """Returns our graph , the bottleneck and the resized input tensor."""
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])  # Setting channels to 3.
        bottleneck_tensor = hub.Module(module_spec)(resized_input_tensor)
    return graph, bottleneck_tensor, resized_input_tensor


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, module_spec):
    """"Perform the necessary ransom transformations. Used for Data Augmentation."""
    # Extract the required image info
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    JPEG_image = tf.image.convert_image_dtype(tf.image.decode_jpeg(jpeg_data, channels=input_depth), tf.float32)  # This is a JPEG representation
    # Scale and crop the image, need to extract the shape and resize the image first.
    scale_value = tf.multiply(tf.constant(1.0 + (random_crop / 100.0)), tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + (random_scale / 100.0)))
    crop_shape = tf.cast(tf.stack([tf.multiply(scale_value, input_height), tf.multiply(scale_value, input_width)]), dtype=tf.int32)
    precropped_image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(JPEG_image, 0), crop_shape), axis=[0])
    cropped_image = tf.random_crop(precropped_image, [input_height, input_width, input_depth])
    # Flip the image if needed.
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image  # Do nothing
    # Change the brightness.
    brightness_value = tf.random_uniform(shape=[], minval=1.0 - (random_brightness / 100.0), maxval=1.0 + (random_brightness / 100.0))
    distort_result = tf.expand_dims(tf.multiply(flipped_image, brightness_value), 0, name='DistortResult')  # Inserts a dimension of 1 into the tensor's shape.
    # Return the placeholder and the distorted tensor.
    return jpeg_data, distort_result


def add_retrain_operations(class_count, final_tensor_name, bottleneck_tensor, is_training):
    """Adds a new softmax layer and configures the optimiser."""

    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        label_input = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')  # Labels

    # Create the weights and biases.
    weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001), name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    # Create the final layer that will be used to classify our images. (y=W*x+b)
    logits = tf.matmul(bottleneck_input, weights) + layer_biases
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # If this is an eval graph, we don't need to add loss ops or optimizer.
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
    """Inserts the operations we need to evaluate the accuracy of our results. Returns an (eval, prediction)"""
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, label_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step) # Make it visible to tensorboard.
    return evaluation_step, prediction


def build_eval_session(module_spec, class_count):
    eval_graph, bottleneck_tensor, resized_input_tensor = create_module_graph(module_spec)
    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        (_, _, bottleneck_input, label_input, final_tensor) = add_retrain_operations(class_count, FLAGS.final_tensor_name, bottleneck_tensor, is_training=False)
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
    """Adds operations that perform JPEG decoding and resizing to the graph"""
    input_height, input_width = hub.get_expected_image_size(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')  # Never evaluate the placeholder directly, always feed it.
    decoded_image_as_float = tf.image.convert_image_dtype(tf.image.decode_jpeg(jpeg_data, channels=hub.get_num_image_channels(module_spec)), tf.float32)
    resize_shape = tf.cast(tf.stack([input_height, input_width]), dtype=tf.int32)  # Cast it as an int
    resized_image = tf.image.resize_bilinear(tf.expand_dims(decoded_image_as_float, 0), resize_shape)
    return jpeg_data, resized_image


def export_model(module_spec, class_count, saved_model_dir):
    """" Saves the model locally"""
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    with sess.graph.as_default():
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name('final_result:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                             legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))
        builder.save()