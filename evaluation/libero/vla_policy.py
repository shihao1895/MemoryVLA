import cv2
import numpy as np
import requests


class LLaVAClient:
    def __init__(self, base_url='http://localhost:6800'):
        self.base_url = base_url

    def reset(self):
        # return requests.post(self.base_url+"/reset").json().get('response')
        pass

    def process_frame(self, text, episode_first_frame, **kwargs):
        state = kwargs.pop('states', None)
        if state is not None:
            state = np.array(state)
            state = state.astype(np.float32)

        encoded_imgs = {}
        for name, img in kwargs.items():
            if img is not None:
                ret, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ret, "Image encode failed"
                encoded_img = encoded_img.tobytes()
            else:
                encoded_img = None
            encoded_imgs.update({name: encoded_img})

        if "image_wrist" in encoded_imgs:
            ret = requests.post(
                self.base_url+"/process_frame",
                data={"text": text, 'episode_first_frame': episode_first_frame,},

                files={"image": encoded_imgs.pop('image', None),
                       "image_wrist": encoded_imgs.pop('image_wrist', None),
                        **({"states": ("states.npy", state.tobytes()) } if state is not None else {}),
                        },
            )
        else:
            ret = requests.post(
                self.base_url + "/process_frame",
                data={"text": text, 'episode_first_frame': episode_first_frame, },

                files={"image": encoded_imgs.pop('image', None),
                       **({"states": ("states.npy", state.tobytes())} if state is not None else {}),
                       },
            )

        return ret.json().get('response')
