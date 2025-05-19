import cv2
import numpy as np
import dynaphos
from dynaphos import utils
from dynaphos import cortex_models
from dynaphos.simulator import GaussianSimulator 
import torch
class PhospheneSimulatorApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BRUSH_SIZE = 5
        self.canvas = None
        self.overlay = None
        self.masks = None
        self.current_index = 0
        self.simulator = self.init_simulator()
        self.IMG_PATHS = ['images/scene_1.png', 'images/scene_2.png', 'images/scene_3.png', 'images/scene_4.png', 'images/scene_5.png']
        self.MASK_PATHS = ['masks/mask_1.npy', 'masks/mask_2.npy', 'masks/mask_3.npy', 'masks/mask_4.npy', 'masks/mask_5.npy']
        self.height, self.width = 512, 512
        self.painting = False

    def init_simulator(self):
        params = utils.load_params('params.yaml')
        coords = cortex_models.get_visual_field_coordinates_probabilistically(params, 750)
        return GaussianSimulator(params, coords)

    def on_mouse_canvas(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.painting = True
            self.draw_circle(x, y, flags)
        elif event == cv2.EVENT_MOUSEMOVE and self.painting:
            self.draw_circle(x, y, flags)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.painting = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.BRUSH_SIZE = min(self.BRUSH_SIZE + 1, 50)
            else:
                self.BRUSH_SIZE = max(self.BRUSH_SIZE - 1, 1)
            cv2.setTrackbarPos('Brush Size', 'Canvas', self.BRUSH_SIZE)

    def draw_circle(self, x, y, flags):
        color = 255 if flags & cv2.EVENT_FLAG_LBUTTON else 0
        cv2.circle(self.canvas, (x, y), self.BRUSH_SIZE, color, -1)

    def on_mouse_overlay(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for mask in self.masks:
                if mask[y, x] > 0:
                    self.canvas[mask > 0] = 255
                    break
        elif event == cv2.EVENT_RBUTTONDOWN:
            for mask in self.masks:
                if mask[y, x] > 0:
                    self.canvas[mask > 0] = 0
                    break

    def on_trackbar_brush(self, val):
        self.BRUSH_SIZE = max(1, val)

    def run(self):
        print(f"Using device: {self.device}")

        self.overlay = cv2.imread(self.IMG_PATHS[self.current_index])
        self.overlay = cv2.resize(self.overlay, (self.width, self.height))
        self.masks = np.load(self.MASK_PATHS[self.current_index])
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        amplitude = 300e-6

        cv2.namedWindow('Overlay')
        cv2.namedWindow('Canvas')
        cv2.namedWindow('Phosphene')
        cv2.moveWindow('Overlay', 0, 200)
        cv2.moveWindow('Canvas', 520, 160)
        cv2.moveWindow('Phosphene', 1040, 200)

        cv2.setMouseCallback('Canvas', lambda *args: self.on_mouse_canvas(*args))
        cv2.setMouseCallback('Overlay', lambda *args: self.on_mouse_overlay(*args))

        cv2.createTrackbar('Brush Size', 'Canvas', self.BRUSH_SIZE, 50, lambda val: self.on_trackbar_brush(val))

        while True:
            stim = self.canvas // 255 * amplitude
            self.simulator.reset()
            stim = cv2.resize(stim, (128, 128))
            stim = self.simulator.sample_stimulus(stim)
            phosphene = self.simulator(stim)
            phosphene = cv2.resize(phosphene.cpu().numpy(), (self.width, self.height))

            cv2.imshow('Overlay', self.overlay)
            cv2.imshow('Canvas', self.canvas)
            cv2.imshow('Phosphene', phosphene)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('n'):
                # Save results
                np.save(f'results/canvas_{self.current_index}.npy', self.canvas)
                cv2.imwrite(f"results/phosphenes_{self.current_index}.jpg", phosphene * 255)
                print(f"Saved canvas/phosphenes_{self.current_index}")

                self.current_index += 1
                if self.current_index < len(self.IMG_PATHS):
                    self.overlay = cv2.imread(self.IMG_PATHS[self.current_index])
                    self.overlay = cv2.resize(self.overlay, (self.width, self.height))
                    self.masks = np.load(self.MASK_PATHS[self.current_index])
                    self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
                else:
                    print("No more images.")
                    break

        cv2.destroyAllWindows()

        
def main():
    app = PhospheneSimulatorApp()
    app.run()

if __name__ == "__main__":
    main()
