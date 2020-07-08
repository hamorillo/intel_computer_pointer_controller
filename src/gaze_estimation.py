'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Model_Gaze_Estimation:
    '''
    Class for the Face Detection Model.
    '''
    infer_times = []

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
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

        # Add extension if it has been specified and we are executing in that device
        if self.cpu_extensions and "CPU" in self.device:
            self.plugin.add_extension(self.cpu_extensions, self.device)

        # Load the IENetwork (model) into the plugin
        start_time = time.time()
        self.exec_network = self.plugin.load_network(self.network, self.device)
        log.info("-> Gaze estimation load time: " +
                 str(time.time()-start_time) + "s")

        return self.plugin

    def predict(self, face, left_eye, right_eye, head_pose_result,
                save_img=False):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_frame, right_eye_frame, pose = self.preprocess_input(
            face, left_eye, right_eye, head_pose_result, save_img)

        self.infer_request = self.exec_network.start_async(
            request_id=0,
            inputs={'left_eye_image': left_eye_frame,
                    'right_eye_image': right_eye_frame,
                    'head_pose_angles': pose})
        return

    def check_model(self):
        if self.infer_request.wait() == 0:
            self.infer_times.append(self.infer_request.latency)
            result = self.infer_request.outputs
            return result

    def preprocess_input(self, face, left_eye, right_eye, head_pose_result,
                         save_img=False):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.exec_network.inputs['left_eye_image'].shape

        width = net_input_shape[3]
        height = net_input_shape[2]

        left_eye_cropped = self.crop_eye(width, height, left_eye, face)
        right_eye_cropped = self.crop_eye(width, height, right_eye, face)

        if(save_img):
            cv2.imwrite("./left_eye_cropped.jpg", left_eye_cropped)
            cv2.imwrite("./right_eye_cropped.jpg", right_eye_cropped)

        left_eye_frame = cv2.resize(
            left_eye_cropped, (net_input_shape[3], net_input_shape[2]))
        left_eye_frame = left_eye_frame.transpose((2, 0, 1))
        left_eye_frame = left_eye_frame.reshape(1, *left_eye_frame.shape)

        right_eye_frame = cv2.resize(
            right_eye_cropped, (net_input_shape[3], net_input_shape[2]))
        right_eye_frame = right_eye_frame.transpose((2, 0, 1))
        right_eye_frame = right_eye_frame.reshape(1, *right_eye_frame.shape)

        angles = [head_pose_result['angle_p_fc'][0][0],
                  head_pose_result['angle_r_fc'][0][0],
                  head_pose_result['angle_y_fc'][0][0]]

        return left_eye_frame, right_eye_frame, angles

    def crop_eye(self, width, height, eye_coord, face):
        face_weight = face.shape[1]
        face_height = face.shape[0]

        x_left = int(eye_coord[0]-(width/2))
        y_left = int(eye_coord[1]-(height/2))
        x_right = int(eye_coord[0]+(width/2))
        y_right = int(eye_coord[1]+(height/2))

        if(x_left < 0):
            x_left = 0

        if(y_left < 0):
            y_left = 0

        if(x_right > face_weight):
            x_right = face_weight

        if(y_right > face_height):
            y_right = face_height

        eye_cropped = face[y_left:y_right, x_left:x_right]
        return eye_cropped

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
