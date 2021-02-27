import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.boxlayout import BoxLayout
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
y = x**2
plt.plot(x,y,label='this is the graph')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

class Figure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super(Figure, self).__init__(plt.gcf(), **kwargs)

class ScreenManagement(ScreenManager):
    pass

class TitleScreen(Screen):
    pass

class AquiferModelling(Screen):
    pass

class RiverEvolution(Screen):
    pass

class GraphScreen(Screen):
    pass

built = Builder.load_file("./ModelProgram.kv")

class ModelProgram(App):
    def build(self):
        return built

if __name__ == '__main__':
    ModelProgram().run()
