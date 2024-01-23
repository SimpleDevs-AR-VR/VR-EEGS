import matplotlib.pyplot as plt
line_types = ["axvline","axhline","axvspan"]

class PlotColor:
  def __init__(self, color, alpha=1.0, linewidth=1.0):
    self.color = color
    self.alpha = alpha
    self.linewidth = linewidth

label_color_map = {
  "SouthSidewalk":PlotColor("mistyrose", alpha=1),
  "NorthSidewalk":PlotColor("mistyrose", alpha=1),
  "RoadCrosswalk":PlotColor("powderblue", alpha=1),
  "RoadWest": PlotColor("black"),
  "RoadEast": PlotColor("black"),

  "SouthSidewalk_Gaze": PlotColor("lightgray",linewidth=2),
  "NorthSidewalk_Gaze": PlotColor("lightgray",linewidth=2),
  "RoadCrosswalk_Gaze": PlotColor("yellow",linewidth=2),
  
  "WalkingSignal": PlotColor("yellow",linewidth=2),
  "CarSignal": PlotColor("yellow",linewidth=2),

  "CommonTree_1": PlotColor("green",linewidth=2),
  "CommonTree_Autumn_2": PlotColor("green",linewidth=2),
  "CommonTree_Autumn_2 (1)": PlotColor("green",linewidth=2),
  "CommonTree_Autumn_2 (2)": PlotColor("green",linewidth=2),
  "CommonTree_Autumn_2 (3)": PlotColor("green",linewidth=2),
  "Tree_Type4_03_mesh": PlotColor("green",linewidth=2),
  "Tree_Type4_03_mesh (1)": PlotColor("green",linewidth=2),

  "Car1": PlotColor("dimgray", linewidth=2),
  "Car2": PlotColor("mediumorchid", linewidth=2),
  "Car3": PlotColor("red",linewidth=2),
  "Car4": PlotColor("seagreen",linewidth=2),
  "Car5": PlotColor("silver",linewidth=2),
  "Jeep2": PlotColor("chartreuse",linewidth=2),
  "Jeep3": PlotColor("black",linewidth=2),
  "Jeep4": PlotColor("darkkhaki",linewidth=2),
  "Jeep5": PlotColor("plum",linewidth=2),
  "MicroBus1": PlotColor("mediumturquoise",linewidth=2),
  "MicroBus2": PlotColor("green",linewidth=2),
  "MicroBus3": PlotColor("oldlace",linewidth=2),
  "MicroBus4": PlotColor("peru",linewidth=2),
  "MicroBus5": PlotColor("ivory",linewidth=2),
  "Sedan1": PlotColor("lightsteelblue",linewidth=2),
  "Sedan2": PlotColor("silver",linewidth=2),
  "Sedan3": PlotColor("black",linewidth=2),

  "SportCar4": PlotColor("black",linewidth=2),
  "SportCar5": PlotColor("maroon",linewidth=2),

  "Truck1": PlotColor("black",linewidth=2)
}

class PlotCondition:
  def __init__(self, 
        conditions, 
        label, 
        line_type="axvline",
        line_width=0.5,
        color="blue", 
        alpha=1):
    self.conditions = conditions
    self.label = label
    self.line_type = line_type
    self.line_width = line_width
    self.color = color
    self.alpha = alpha
  def SetColor(self, color):
    self.color = color
    return self
  def SetAlpha(self, alpha):
    self.alpha = alpha
    return self
  def SetType(self, t):
    if t not in line_types: print("ERROR: cannot set line type {t}")
    else: self.line_type = t
    return self
  def SetWidth(self, w):
    self.line_width = w
    return self
  def GetConditions(self):
    return " & ".join(self.conditions)
  def AddConditions(self, new_conditions):
    self.conditions.extend(new_conditions)
    return self


"""
============================
SIMULATION EVENTS
============================
"""
trial_start = PlotCondition(
  conditions = ['(event_type == "Simulation")'], 
  label = "Trials", 
  color = "black", 
  alpha = 0.5
)



"""
============================
PLAYER POSITION CONDITIONS
============================
"""
pos_north = PlotCondition(
  conditions = ['(event_type == "Player")','(title == "position")', '(description == "NorthSidewalk")'], 
  label = "Player Pos: North Sidewalk", 
  color = "blue", 
  alpha = 0.05,
  line_type = "axvspan"
)
pos_south = PlotCondition(
  conditions = ['(event_type == "Player")','(title == "position")','(description == "SouthSidewalk")'],
  label = "Player Pos: South Sidewalk",
  color = "red",
  alpha = 0.05,
  line_type = "axvspan"
)
pos_crosswalk = PlotCondition(
  conditions = ['(event_type == "Player")','(title == "position")','(description == "RoadCrosswalk")'],
  label = "Player Pos: Crosswalk",
  color = "black",
  alpha = 0.05,
  line_type = "axvspan"
)


"""
============================
GAZE TARGET CONDITIONS
============================
"""

#### --- ALL ELEMENTS --- #####
all_targets = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")'
  ], label = "Player Gaze: All elements"
)


##### --- ENVIRONMENT --- #####
walking_signals = PlotCondition(
  conditions = [
    '(event_type == "Global Eye Tracking")',
    '(title == "Left")',
    '((description=="WalkingSignal") | (description=="WalkingSignalCollider"))'
  ], label = "Player Gaze: Walking Signal",
  color = "gold",
  line_width = 3
)
car_signals = PlotCondition(
  conditions = [
    '(event_type == "Global Eye Tracking")',
    '(title == "Left")',
    '((description=="CarSignal") | (description=="WalkingSignalCollider"))'
  ], label = "Player Gaze: Car Signal",
  color = "blue",
  line_width = 3
)


##### --- CARS --- #####
car1 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Car1")'
  ], label = "Player Gaze: Car1",
  color = "dimgray"
)
car2 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Car2")'
  ], label = "Player Gaze: Car2",
  color = "mediumorchid"
)
car3 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Car3")'
  ], label = "Player Gaze: Car3",
  color = "red"
)
car4 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Car4")'
  ], label = "Player Gaze: Car4"
)
car5 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Car5")'
  ], label = "Player Gaze: Car5",
  color = "silver"
)

##### --- JEEPS --- #####
jeep2 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Jeep2")'
  ], label = "Player Gaze: Jeep2",
  color = "chartreuse"
)
jeep3 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Jeep3")'
  ], label = "Player Gaze: Jeep3",
  color = "black"
)
jeep4 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Jeep4")'
  ], label = "Player Gaze: Jeep4",
  color = "darkkhaki"
)
jeep5 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Jeep5")'
  ], label = "Player Gaze: Jeep5",
  color = "plum"
)

##### --- MICROBUSES --- #####
microbus1 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "MicroBus1")'
  ], label = "Player Gaze: MicroBus1",
  color = "mediumturquoise"
)
microbus2 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "MicroBus2")'
  ], label = "Player Gaze: MicroBus2",
  color = "green"
)
microbus3 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "MicroBus3")'
  ], label = "Player Gaze: MicroBus3",
  color = "oldlace"
)
microbus4 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "MicroBus4")'
  ], label = "Player Gaze: MicroBus4",
  color = "peru"
)
microbus5 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "MicroBus5")'
  ], label = "Player Gaze: MicroBus5",
  color = "ivory"
)

##### --- SEDANS --- #####
sedan1 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Sedan1")'
  ], label = "Player Gaze: Sedan1",
  color = "aquamarine"
)
sedan2 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Sedan2")'
  ], label = "Player Gaze: Sedan2",
  color = "lightsteelblue"
)
sedan3 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Sedan3")'
  ], label = "Player Gaze: Sedan3",
  color = "black"
)
sedan4 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Sedan4")'
  ], label = "Player Gaze: Sedan4"
)
sedan5 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Sedan5")'
  ], label = "Player Gaze: Sedan5"
)

##### --- SPORTS --- #####
sport4 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "SportCar4")'
  ], label = "Player Gaze: SportCar4",
  color = "black"
)
sport5 = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "SportCar5")'
  ], label = "Player Gaze: SportCar5",
  color = "maroon"
)




##### --- TRUCKS --- #####
truck = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "Truck1")'
  ], label = "Player Gaze: Truck1",
  color = "lightpink"
)


##### --- ENVIRONMENTAL ELEMENTS --- #####
car_signal = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "CarSignal")'
  ], label = "Player Gaze: CarSignal"
)
walking_signal = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "WalkingSignal")'
  ], label = "Player Gaze: WalkingSignal"
)
north_sidewalk = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "NorthSidewalk")'
  ], label = "Player Gaze: North Sidewalk"
)
south_sidewalk = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "SouthSidewalk")'
  ], label = "Player Gaze: South Sidewalk"
)
road_crosswalk = PlotCondition(
  conditions = [
      '(event_type == "Global Eye Tracking")',
      '(title == "Left")',
      '(description == "RoadCrosswalk")'
  ], label = "Player Gaze: Road Crosswalk"
)
