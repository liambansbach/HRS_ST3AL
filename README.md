# HRS_ST3AL

## Running the upper body imitation pipeline

This tutorial guides you through the necessary setup in order to run the upper body imitation on the Ainex Robot.

### Prerequisites

Before starting this tutorial, ensure you have a working ROS2 environment installed.

## Step 1: Environment Setup and Build

The **MediaPipe** library does not come packaged with ROS2 and need to be installed separately (preferably in a virtual environment). In this step, we will set up a virtual environment, install dependencies, and build the ROS2 workspace.

1.  **Create and Activate Virtual Environment:**
    Navigate to your workspace `src` folder and create the environment:
    ```bash
    cd src
    python3 -m venv .venv --system-site-packages --symlinks
    source .venv/bin/activate
    ```


    **WARNING:** It is important that the virtual environment is placed correctly inside "src" and is named ".venv". If not, the launch description will not find it!

2.  **Install MediaPipe:**
    With the virtual environment active, install the required libraries:
    ```bash
    pip install mediapipe pin casadi "numpy<2"
    ```

3.  **Build the Workspace:**
    Return to the root of your workspace and build the packages:
    ```bash
    cd ..
    colcon build
    ```

4.  **Source the Setup Script:**
    Overlay the workspace environment:
    ```bash
    source install/setup.bash
    ```

## Step 2: Run the launch file

Run the launch file **from the root of the project**:

```bash
ros2 launch bringup upper_body_imitation.launch.py
```

## Running the cube stacking pipeline

### Prerequisites

Before starting, make sure you have a working **ROS 2** environment.

The only external dependency required for this pipeline is **Pinocchio**, installed system-wide via **robotpkg**.

---

### Install Pinocchio (robotpkg)

If you already have **robotpkg** configured, skip to **Step 5**.

#### 1) Install required tools
```bash
sudo apt install -qqy lsb-release curl
```

#### 2) Register the robotpkg signing key

```bash
sudo mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
  | sudo tee /etc/apt/keyrings/robotpkg.asc
```

#### 3) Add robotpkg as an APT repository

```bash
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
  | sudo tee /etc/apt/sources.list.d/robotpkg.list
```

#### 4) Update package list

```bash
sudo apt update
```

#### 5) Install Pinocchio

```bash
sudo apt install -qqy robotpkg-py3*-pinocchio
```

This installs Pinocchio and all required dependencies.

---

### Run the pipeline

1. Build your workspace:

```bash
colcon build
```

2. Source the workspace:

```bash
source install/setup.bash
```

3. Launch the stacking pipeline:

```bash
ros2 launch bringup stacking_pipeline.launch.py gui:=false rviz:=false
```

## Run the whole setup using a state machine.

