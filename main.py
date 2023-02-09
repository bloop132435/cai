from kivy.app import App
from kivy.uix.label import Label
import numpy as np
from kivy.config import Config
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.image import Image

from constants import *
from perceptron import Perceptron
from shape import square,circle
from utils import save_heatmap
import trainer

Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '500')


class Screen(FloatLayout):
    '''
    by Geoffrey Qian
    The class that contains the application logic
    '''
    def __init__(self, **kwargs):
        '''Constructor for the app'''
        super(Screen, self).__init__(**kwargs)
        size_hint = (0.1,0.05)
        self.slider = Slider(orientation='vertical', min=0, max=SLIDER_RES, on_touch_move=self.touch_move,
                             on_touch_up=self.update_shape, pos_hint={'x': 0, 'y': 0}, size_hint=(0.1, 1),sensitivity='handle')
        self.add_widget(self.slider)
        self.square = ToggleButton(text='Square', state='down', group='shape', size_hint=(
            0.1, 0.05), pos_hint={'x': 0.1, 'y': 0},on_release=self.update_shape)
        self.circle = ToggleButton(text='Circle', group='shape', size_hint=(
            0.1, 0.05), pos_hint={'x': 0.2, 'y': 0},on_release=self.update_shape)
        self.add_widget(self.square)
        self.add_widget(self.circle)
        self.train_custom_btn = Button(text="Train custom",size_hint=size_hint,pos_hint={'x':0.4,'y':0},on_press=self.train_custom)
        self.train_btn = Button(text="Train 20",size_hint=size_hint,pos_hint={'x':0.5,'y':0},on_press=self.train_20)
        self.autotrain_btn = Button(text="Auto-train",size_hint=size_hint,pos_hint={'x':0.6,'y':0},on_press=self.autotrain)
        self.test_custom_btn = Button(text="Test custom",size_hint=size_hint,pos_hint={'x':0.7,'y':0},on_press=self.test_custom)
        self.autotest_btn = Button(text="Test 100",size_hint=size_hint,pos_hint={'x':0.8,'y':0},on_press=self.autotest)
        self.add_widget(self.train_custom_btn)
        self.add_widget(self.train_btn)
        self.add_widget(self.autotrain_btn)
        self.add_widget(self.test_custom_btn)
        self.add_widget(self.autotest_btn)
        self.model = Perceptron(np.float32(
            LEARNING), weight_decay=np.float32(WEIGHT))

        self.input = Image(source="./input.png",size_hint=(0.4,0.7),pos_hint={'x':0.15,'y':0.3},allow_stretch=True,keep_ratio=True)
        self.weight = Image(source="./weight.png",size_hint=(0.4,0.7),pos_hint={'x':0.6,'y':0.3},allow_stretch=True,keep_ratio=True)
        self.add_widget(self.input)
        self.add_widget(self.weight)

        self.accuracy = Label(text="Accuracy: NA%",size_hint=size_hint,pos_hint={'x':0.6,'y':0.1})
        self.num_data = 0
        self.data_tested = Label(text="# of Training Images: 0",size_hint=size_hint,pos_hint={'x':0.6,'y':0.2})
        self.add_widget(self.accuracy)
        self.add_widget(self.data_tested)

    def update_label(self):
        '''updates the training images label'''
        self.data_tested.text = f"# of Training Images: {self.num_data}"
        pass

    def train(self,num:int):
        '''train the model on num pairs of images'''
        trainer.train(self.model,num)
        self.num_data += 2 * num
        self.update_label()
        save_heatmap(self.model.hidden_layer,"weight.png")
        self.weight.reload()


    def train_20(self, *_):
        '''train the model on 20 images'''
        self.train(10)
        acc = trainer.test(self.model,20)
        self.accuracy.text = f"Accuracy: {acc * 100}%"
        pass

    def autotrain(self, *_):
        '''auto train the model until it reaches 95% accuracy'''
        while True:
            acc = trainer.test(self.model,20)
            if acc > 0.95:
                self.accuracy.text = f"Accuracy: {acc * 100}%"
                return
            self.train_20()

    def autotest(self, *_):
        '''tests the model on 100 images'''
        acc = trainer.test(self.model,100)
        self.accuracy.text = f"Accuracy: {acc * 100}%"

    def test_custom(self,  *_):
        '''test the model on the user input image'''
        size = int(IMAGE_SIZE//2 * self.slider.value_normalized)
        arr = np.array([])
        if self.square.state == 'down':
            arr = square((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        else:
            arr = circle((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        result = self.model.forward(arr)
        correct = ~(result ^ (self.square.state == 'down'))
        if correct:
            self.accuracy.text = f"Accuracy: 100%"
        else:
            self.accuracy.text = f"Accuracy: 0%"

    def train_custom(self, *_):
        '''train the model on the user input image'''
        size = int(IMAGE_SIZE//2 * self.slider.value_normalized)
        arr = np.array([])
        if self.square.state == 'down':
            arr = square((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        else:
            arr = circle((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        result = self.model.forward(arr)
        correct = ~(result ^ (self.square.state == 'down'))
        if correct:
            self.accuracy.text = f"Accuracy: 100%"
        else:
            self.accuracy.text = f"Accuracy: 0%"
        self.model.backward(arr,self.square.state=='down',result)
        self.num_data += 1
        self.update_label()
        save_heatmap(self.model.hidden_layer,"weight.png")
        self.weight.reload()






    def update_shape(self, *_):
        '''update the shape in the application display when the slider is released'''
        size = int(IMAGE_SIZE//2 * self.slider.value_normalized)
        arr = np.array([])
        if self.square.state == 'down':
            arr = square((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        else:
            arr = circle((IMAGE_SIZE//2,IMAGE_SIZE//2),size)
        save_heatmap(arr,"input.png")
        self.input.reload()

    def touch_move(self, *touch):
        '''touch handler for the slider widget'''
        if touch[1].sx >= 0.1:
            return True
        self.slider.value = int(SLIDER_RES * touch[1].sy)
        return True


class myApp(App):
    '''
    by Geoffrey Qian
    the wrapper class that calls the application code
    '''
    def build(self):
        return Screen()


if __name__ == '__main__':
    blank = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
    save_heatmap(blank,"input.png")
    save_heatmap(blank,"weight.png")
    myApp().run()
