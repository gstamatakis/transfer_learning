from parser_options import *
from utils import *


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:  # The .pb graph
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    file_reader = tf.read_file(file_name, "file_reader")
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = tf.Session().run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    graph = load_graph(testFLAGS.graph)

    input_operation = graph.get_operation_by_name("import/" + testFLAGS.input_layer)
    output_operation = graph.get_operation_by_name("import/" + testFLAGS.output_layer)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: read_tensor_from_image_file(testFLAGS.image, input_height=testFLAGS.input_height, input_width=testFLAGS.input_width, input_mean=testFLAGS.input_mean, input_std=testFLAGS.input_std)
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]  # Print only the top 5
    labels = load_labels(testFLAGS.labels)
    for i in top_k:
        print(labels[i], results[i])
