# To Do List
## Critical Features
- Error handling
	- Preventing 'death loop' (motor moving in a single direction, increasing the error permanently)
	- Handling not all beams being present on the camera

## Non-Critical Features
- Logging the motor movements into a .csv file
- Interactive / automatic assignment of the spots / motor axes
	- Either by a user interface (buttons to move a motor a little, look at how the spots change and then assign an axis / spot)
	- Or by doing something automatically (the program itself iterates through the motors to figure out how to assign both the axis and spot)
- Adding 'I' (and possibly 'D') to the PID control
	- The code seems to work well enough with just 'P', but 'I' and 'D' may improve things further and make the system a little more stable
- Code cleanup / comments
- Speed optimisation