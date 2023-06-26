# SpheroidSizer
Functions to detect Spheroids automaticly or to draw the outlines maually if automatic detection fails

### Set up
- make sure that the packages *numpy*, *pandas*, *tinker* and *cv2* are installed on your machine

```ruby
pip install numpy
pip install pandas
pip install tkinter
pip install opencv-python
```

- Save images for one experiment in one folder. Use for each day of the experiment a subfolder, if you want to seperate them

### Usage
- Run SpheroidSizer.py
- A window will open where ypu habe to choose your main folder
- The images will be analysed, one after the other:
  - Automatic detected Spheroid will be displayed
  - Accept (ENTER) or Start manuel drawing (M)
  - For manuel drawing a new window opends. Left klick to ad point, right click to remove to last one, ENTER when finished.

### Analysis
A adapeted script from https://github.com/DaRoemer/Spheroid is added (Spheroid_fct.py). This funcions enable one to a fast analysis of the data. Mainly it helps to find the most **repressentativ** Spheroid in a experiment with multiple replicates over multiple days.


I hope this helps. Fell free to adapt the code and use whatever you want. The goal of this project was mainly to get used with cv2 while solving a present lab problem.
Have fun :v:
