import caffe
import numpy as np
from scipy.io import loadmat

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
ref_model_file = '/home/ubuntu/machinevision/caffe/models/bvlc_googlenet/deploy.prototxt'
ref_pretrained = '/home/ubuntu/machinevision/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

caffe.set_mode_cpu()
imagenet_mean = np.load('/home/ubuntu/machinevision/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
net = caffe.Classifier(ref_model_file, ref_pretrained,
                       mean=imagenet_mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

image_file = '/path/to/some/image.jpg'
input_image = caffe.io.load_image(image_file)

# Call the prediction
output = net.predict([input_image])
predictions = output[0]

# Find the top three predictions
predicted_class_index = predictions.argmax()
ind = np.argpartition(predictions, -3)[-3:]

# Format HTML to return (Unless running in Exaptive,
# you probably want to pretty print this to the console.)
retstr = "<h3>GoogLeNet:</h3>"
for i in range(0, len(ind[np.argsort(predictions[ind])])):
    retstr += "#%d. %s (%2.1f%%) <br>" % (i + 1, name_map[ind[np.argsort(predictions[ind])][2-i]], predictions[ind[np.argsort(predictions[ind])][2-i]] * 100)

return retstr
