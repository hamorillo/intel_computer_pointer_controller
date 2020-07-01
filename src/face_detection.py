'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore


class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.cpu_extensions = extensions

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model_name + ".xml"
        model_bin = self.model_name + ".bin"

        # Inference Engine initialization
        self.plugin = IECore()

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        supported_layers = self.plugin.query_network(
            network=self.network, device_name="CPU")

        unsupported_layers = [
            l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Add extension if it has been specified and we are executing in that device
        if self.cpu_extensions and "CPU" in self.device:
            self.plugin.add_extension(self.cpu_extensions, self.device)

        # Load the IENetwork (model) into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.plugin

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_image = self.preprocess_input(image)

        self.infer_request = self.exec_network.start_async(
            request_id=0, inputs={self.input_blob: p_image})
        return

    def check_model(self):
        if self.infer_request.wait() == 0:
            result = self.infer_request.outputs[self.output_blob]
            return result

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.exec_network.inputs[self.input_blob].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, frame, threshold, save_img=False):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        height = frame.shape[0]
        width = frame.shape[1]
        face_boxes = []
        for box in outputs[0][0]:  # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                face_boxes.append([xmin, ymin, xmax, ymax])

                if(save_img):
                    cv2.rectangle(frame, (xmin, ymin),
                                  (xmax, ymax), (0, 0, 255), 1)

                    cv2.imwrite("./output.jpg", frame)
        return frame, face_boxes
