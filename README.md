

## Package Info: Tic-Tac-Toe Project

[![license - apache 2.0](https://img.shields.io/:license-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

The Tic-Tac-Toe project is a sample package for using a 6-axis robotic arm to play against a human in a game of tic-tac-toe. Project was originally a simple cube pickup which is documented in this repo. Intended to support the development of robotic applications at the Artifically Intelligent Manufacturing Systems (AIMS) Lab at The Ohio State University.

Note: This repository was designed for ROS Melodic. It has not been tested on other distributions. Specifically designed for the Motoman MH5L robot as supported by the ROS-Industrial program.


### Prerequisites
  - **ROS Melodic:** For obtaining and configuring ROS follow the installation instructions for [full-desktop Melodic installation](http://wiki.ros.org/melodic/Installation/Ubuntu).
  - **Catkin workspace:** Create a clean [catkin-tools](https://catkin-tools.readthedocs.io/en/latest/index.html) workspace.
  - **MoveIt 1:** For installation instructions see [MoveIt's webpage](https://moveit.ros.org/install/).
 
### Required Repositories
  Clone the below repositories into your catkin-tools workspace:
  - [tic-tac-toe](https://github.com/osu-aims/tic-tac-toe)
  - [aims-robots-all](https://github.com/osu-aims/aims-robots-all)

## Installation
  1. Create a catkin compatible workspace ```$ mkdir -p catkin_ws/src```
  2. Clone the repositories listed above into the src folder
  3. Build and source workspace 
     ```
     catkin init
     catkin build
     source devel/setup.bash
     ```
     
Once the workspace is built, you are ready to start executing commands.

#### Install missing dependencies
If the build fails, it occurs usually to missing package dependencies or missing third party (non-ros) packages. When this occurs the build log in the terminal indicates the name of the package dependency that it is missing, then try:

```
sudo apt-get update ros-melodic-[package-name]
# separate the package name words with a '-'
```
If a package is not found it is probably a third-party dependency, google the name of the package and search for installation instructions:

## Usage

```
roslaunch tictactoe tictactoe.launch
rosrun tictactoe tictactoe.py
```




