import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import numpy as np
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

plt.plot([1,2,6,3])

class MenuScreen(Screen):
    pass
class ValueScreen(Screen):
    pass
class MapScreen(Screen):
    #plot = plt
    #fig = FigureCanvasKivyAgg(figure = plt.gcf(),id='mapFigure')
    '''
    def __init__(self, **kwargs):
        #self.ids['figureBoxLayout'].add_widget(FigureCanvasKivyAgg(plt.gcf()))
        self.name = 'map'
    '''
    pass

class EarthScience(App):
    def build(self):
        # Create screen
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(ValueScreen(name='value'))
        sm.add_widget(MapScreen(name='map'))

        #box = BoxLayout()
        #box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        #box.add_widget(Label(text="This is some text"))
        #return box

        return sm

if __name__ == '__main__':
    EarthScience().run()
