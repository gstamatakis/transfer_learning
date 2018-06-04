from datetime import datetime

from utils import *

# Disables AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    # Change to INFO later on
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Prepare necessary directories that can be used during training
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0 and not os.path.exists(FLAGS.intermediate_output_graphs_dir):
        os.makedirs(FLAGS.intermediate_output_graphs_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())

    # See if the command-line flags mean we're applying any distortions.
    distoreted_images = FLAGS.flip_left_right or FLAGS.random_crop != 0 or FLAGS.random_scale != 0 or FLAGS.random_brightness != 0

    # URL of the pre-trained graph.
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor = create_module_graph(module_spec)

    # Add the new layer that we'll be training.
    with graph.as_default():
        train_step, cross_entropy, bottleneck_input, labels_input, final_tensor = add_final_retrain_ops(class_count, FLAGS.final_tensor_name, bottleneck_tensor, is_training=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Used for flexible memory management
    config.gpu_options.per_process_gpu_memory_fraction = 0.95  # GPU tf is allowed to use.
    config.gpu_options.allocator_type = 'BFC'

    with tf.Session(graph=graph, config=config) as sess:
        # Initialize all weights: for the module to their pretrained values and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        if distoreted_images:
            # We will be applying distortions, so set up the operations we'll need.
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness, module_spec)
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and cached them on disk.
            bottleneck_cnt = 0
            if not os.path.exists(FLAGS.bottleneck_dir):
                os.makedirs(FLAGS.bottleneck_dir)
            for label_name, label_lists in image_lists.items():
                for category in ['training', 'testing', 'validation']:
                    category_list = label_lists[category]
                    for index, _ in enumerate(category_list):
                        get_or_create_bottleneck(sess, image_lists, label_name, index, FLAGS.image_dir, category, FLAGS.bottleneck_dir, jpeg_data_tensor,
                                                 decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)
                        bottleneck_cnt += 1
                        if bottleneck_cnt % 1000 == 0:
                            tf.logging.info(str(bottleneck_cnt) + ' bottleneck files created.')
                tf.logging.info('A total of ' + str(bottleneck_cnt) + ' bottleneck files were created.')

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, labels_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph when exporting models.
        train_saver = tf.train.Saver()

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            try:
                # Get a batch of input bottleneck values, either calculated fresh every time with distortions applied, or from the cache stored on disk.
                if distoreted_images:
                    (train_bottlenecks, train_labels) = get_random_distorted_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training', FLAGS.image_dir, distorted_jpeg_data_tensor, distorted_image_tensor,
                                                                                         resized_image_tensor, bottleneck_tensor)
                else:
                    (train_bottlenecks, train_labels, _) = get_random_cached_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training', FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                                                                                         decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)

                # Feed the bottlenecks and ground truth into the graph, and run a training step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks, labels_input: train_labels})
                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                if (i % FLAGS.eval_step_interval) == 0 or i + 1 == FLAGS.how_many_training_steps:
                    train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, labels_input: train_labels})
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))

                    # moving averages being updated by the validation set, though in practice this makes a negligible difference.
                    validation_bottlenecks, validation_labels, _ = (get_random_cached_bottlenecks(sess, image_lists, FLAGS.validation_batch_size, 'validation', FLAGS.bottleneck_dir, FLAGS.image_dir,
                                                                                                  jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module))

                    # Run a validation step and capture training summaries for TensorBoard with the `merged` op.
                    validation_summary, validation_accuracy = sess.run([merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, labels_input: validation_labels})
                    validation_writer.add_summary(validation_summary, i)
                    tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))
            except KeyboardInterrupt:
                tf.logging.info('%s: Keyboard Interrupt, will stop training')
                break

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0:
                # If we want to do an intermediate save, save a checkpoint of the train graph, to restore into the eval graph.
                train_saver.save(sess, FLAGS.checkpoint_path)
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
                save_graph_to_file(intermediate_file_name, module_spec, class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, FLAGS.checkpoint_path)

        # Run a final test evaluation on some new images we havent used before.
        test_bottlenecks, test_labels, test_filenames = (
            get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size, 'testing', FLAGS.bottleneck_dir,
                                          FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module))

        (eval_session, _, bottleneck_input, labels_input, evaluation_step, prediction) = build_eval_session(module_spec, class_count)
        test_accuracy, predictions = eval_session.run([evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, labels_input: test_labels})
        tf.logging.info('Test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        # Write out the trained graph and labels with the weights stored as constants.
        tf.logging.info('Final result saved to : ' + FLAGS.output_graph)
        save_graph_to_file(FLAGS.output_graph, module_spec, class_count)
        with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if FLAGS.saved_model_dir:
            export_model(module_spec, class_count, FLAGS.saved_model_dir)


if __name__ == '__main__':
    tf.app.run(main=main)
