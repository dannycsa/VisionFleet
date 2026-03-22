# VisionFleet

## Introduction
VisionFleet is a collaborative mapping system designed to detect and report temporary urban traffic events, such as roadblocks or unannounced construction, in real-time. Traditional navigation systems often have a delay in reflecting these disruptions. This project proposes using a fleet of micromobility vehicles equipped with standard cameras and GPS receivers to map these events collectively.

The system operates on the CARLA simulator and uses ROS 2 to manage communication. Instead of relying entirely on perfect individual sensors, it uses a multi-agent consensus approach. If multiple vehicles independently detect an obstacle in the same location, the system confirms the event. Conversely, if a vehicle drives through a reported area without detecting anything, the system lowers the confidence of that obstacle and eventually removes it. This method effectively filters out false alarms without needing expensive hardware.

## System Components
The architecture is divided into independent modules:
* **Fleet Manager:** Spawns and controls the autonomous driving of the bicycles within the simulation.
* **Perception:** Processes camera feeds using a YOLOv8 model combined with color filters to identify physical blockades and ignore false positives like road lines.
* **Sensor Fusion:** Translates raw GPS data into a usable metric map and calculates vehicle orientation.
* **Consensus Engine:** The core logic that groups detections, applies multi-agent quorum rules, and updates the map based on what the fleet sees (or doesn't see).
* **Graphical Interface:** A real-time dashboard displaying the map, vehicle positions, and the confidence levels of detected obstacles.

## How to Run

### 1. Start the Simulator
First, launch the CARLA environment. Running it with low quality and off-screen rendering helps maintain stability and performance.
```bash
./CarlaUE4.sh --ros2 -quality-level=Low -RenderOffScreen
```
### 2. Launch the Infrastructure
Once the simulator is active, start the background processes. This will initialize the data fusion, the consensus map, and the visual interface.

```bash
ros2 launch visionfleet_biker visionfleet_infra.launch.py
```

### 3. Deploy the Fleet
Finally, spawn the bicycles into the map to begin the session. You can define how many vehicles you want to run simultaneously.
```bash
ros2 launch visionfleet_biker visionfleet_session.launch.py quantity:=4
```
