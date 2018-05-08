#!/bin/sh
python image_retraining.py \
--image_dir  /content/flower_photos  \
--output_graph  /content/e1/output_graph.pb  \
--intermediate_output_graphs_dir  /content/e1/intermediate_graph  \
--output_labels  /content/e1/output_labels.txt  \
--summaries_dir  /content/e1/retrain_logs  \
--bottleneck_dir  /content/e1/bottleneck  \
--checkpoint_path  /content/e1/checkpoints  \
--tfhub_module  https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1  \
--image  /content/flower_photos/daisy/5547758_eea9edfd54_n.jpg  \
--input_layer  input  \
--final_tensor_name  final_result  \
--output_lay  InceptionV3/Predictions/Reshape_1  \
--graph  /content/e1/output_graph.pb  \
--labels  /content/e1/output_labels.txt  \
--optimizer  adam  \
--min_images_per_label 20 \
--max_images_per_label 1000000 \
--how_many_training_steps 4000 \
--intermediate_store_frequency 500 \
--learning_rate 0.001 \
--beta1 0.9 \
--beta2 0.999 \
--epsilogn 0.00000001 \
--decay 0.9 \
--momentum 0.0 \
--testing_percentage 10 \
--validation_percentage 10 \
--eval_step_interval 10 \
--train_batch_size 32 \
--test_batch_size -1 \
--validation_batch_size 128 \
--random_crop 0 \
--random_scale 0 \
--random_brightness 0 \
--input_height 299 \
--input_width 299 \
--input_mean 0 \
--input_std 255 \
