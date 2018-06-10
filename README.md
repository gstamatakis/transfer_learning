#Project
Transfer learning and Data augmentation using Tensorflow.           
Giorgos Stamatakis - 2013 030 154               
Giorgos Dialektatkis - 2014 030 178         

#Running/Deploying the model
Scrips are placed in the code directory.
The report ,along with some experiment, graphs is placed in the docs folder.
To deploy (or use in an ipython notebook) consult the book_cmds in the docs folder.

#Creating and training the model
Transfer learning scripts that use TensorFlow to train a neural network by reusing parts 
of another bigger NN trained on different labels.

Check parser options for a list of available params or type python transfer_learning.py -h.
All params are set to some default values but some must be changed as they are OS or FS specific.

You need to set the dataset path and the output paths for a few different things (labels,graph,summaries...)
Simply replace the "E:\tf_proj1" part in every arg with the desired path in your FS.

#Dataset
Flowers dataset can be found here: 
http://download.tensorflow.org/example_images/flower_photos.tgz 

#Testing the model
After training the model you can use: python label_image.py --image <image_path> --graph <.pb graph path> 
to test the model.You can also set some other params which again can be found in the bottom of 
the parser_options.py file.