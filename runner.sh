#!/bin/sh
python image_retraining.py \
--image_dir='/tmp/flower_photos' \
--output_graph '/tmp/e1/output_graph.pb' \
--intermediate_output_graphs_dir '/tmp/e1/intermediate_graph' \
--output_labels '/tmp/e1/output_labels.txt' \
--summaries_dir '/tmp/e1/retrain_logs' \
--bottleneck_dir '/tmp/e1/bottleneck' \
--checkpoint_path '/tmp/e1/checkpoints' \
--tfhub_module 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1' \
--image '/tmp/flower_photos/daisy/5547758_eea9edfd54_n.jpg' \
--input_layer 'input' \
--final_tensor_name 'final_result' \
--output_lay 'InceptionV3/Predictions/Reshape_1' \
--graph '/tmp/e1/output_graph.pb' \
--labels '/tmp/e1/output_labels.txt' \
--optimizer 'adam' \
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
