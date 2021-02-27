from tkinter import *

class App:
    def __init__(self):
        self.app = Tk()
        self.title_screen = TitleScreen(self.app, self)
        self.hydrogeology = Hydrogeology(self.app)


    def run(self):
        self.app.mainloop()

    # Switching between screens
    def switch_to_hydrogeology(self):
        self.title_screen.destroy()
        self.hydrogeology.show()
        print("Lifted hydrogeology")


# Creating the frames
# The main frame
class TitleScreen(Frame):
    def __init__(self, master=None, root = None):
        super().__init__(master)
        self.master = master
        self.root = root
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Title
        self.title = Label(
            master = self,
            text = "Earth Science Modelling Programs",
            width = 30,
            height = 2,
            font = ("Arial",18)
        )
        self.title.grid(row = 0, column = 0)
        # Hydrogeology
        self.hydrogeology_button = Button(
            master = self,
            text = "Aquifer Modelling",
            font = ("Arial",18),
            relief = RAISED,
            width = 30,
            bd = 7,
            command = self.root.switch_to_hydrogeology
        )
        self.hydrogeology_button.grid(row = 1, column = 0)



        # Hydrogeology
        self.river_evolution_button = Button(
            master = self,
            text = "River Evolution",
            font = ("Arial",18),
            relief = RAISED,
            width = 30,
            bd = 7,
            command = do_nothing_river_evolution
        )
        self.river_evolution_button.grid(row = 2, column = 0)

    def switch_to_hydrogeology(self):
        pass

    def show(self):
        self.lift()



class Hydrogeology(Frame):
    def __init__self(master = None):
        super().__init__(master)
        self.master = master
        self.grid(row = 0, column = 0)
        self.create_widgets()

    def create_widgets(self):
        # Title
        self.title = Label(
            master = self,
            text = "Earth Science Modelling Programs",
            width = 30,
            height = 2,
            font = ("Arial",18)
        )
        self.title.grid(row = 0, column = 0)

        #
        print("have not yet done it")

    def show(self):
        self.lift()

    def make_main(self):
        self.grid(row = 0, column = 0)



# Some functions
def do_nothing_hydro():
    print("'Aquifer modelling' has not been implemented yet")

def do_nothing_river_evolution():
    print("'River Evolution' has not been implemented yet")


app = App()
app.run()
