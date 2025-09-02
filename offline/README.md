Offline setup overview

1) Build wheels on an online machine matching the target OS/arch/Python.
   - Create a folder offline/wheels/ and pre-download all .whl files.
   - Example to vendor a dependency:
     pip download --only-binary=:all: -d offline/wheels numpy==1.26.4

2) Include torch CPU wheel matching platform, and a wheel for segment-anything
   (build from source once online; place the .whl into offline/wheels/).

3) Generate a pinned requirements list (requirements_offline.txt) and install offline via:
   pip install --no-index --find-links=offline/wheels -r offline/requirements_offline.txt

4) Use start_wafer_tool.command / start_wafer_tool_windows.cmd. The launcher
   can be modified to prefer the offline wheels path by setting PIP_FIND_LINKS
   and PIP_NO_INDEX in the environment before running check_and_launch.py.

