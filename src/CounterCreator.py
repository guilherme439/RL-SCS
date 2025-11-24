from .SCS_Renderer import SCS_Renderer
from ._utils import get_package_root

class CounterCreator:

    def __init__(self):
        self.package_root = get_package_root()
        return
    
    def new_counter(self, option):

        match option:
            case 1:               

                renderer = SCS_Renderer()

                renderer.create_counter_from_scratch("ally_soldier", (1,2,2), "infantary", color_str="dark_green")
                renderer.add_border("green", str(self.package_root / "assets" / "ally_soldier.jpg"))

                renderer.create_counter_from_scratch("ally_tank", (2,2,4), "mechanized", color_str="dark_green")
                renderer.add_border("green", str(self.package_root / "assets" / "ally_tank.jpg"))

                renderer.create_counter_from_scratch("axis_soldier", (1,1,3), "infantary", color_str="dark_red")
                renderer.add_border("red", str(self.package_root / "assets" / "axis_soldier.jpg"))

                renderer.create_counter_from_scratch("axis_tank", (4,6,1), "mechanized", color_str="dark_red")
                renderer.add_border("red", str(self.package_root / "assets" / "axis_tank.jpg"))


                print("\nImages created!\n")
