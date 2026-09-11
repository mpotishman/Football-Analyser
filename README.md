# Football Analyser: Player Tracking and Shirt Number Recognition From Match Footage

⚽ A computer vision project that takes football footage, finds the useful parts of the broadcast, tracks the players and ball, separates the two teams by kit colour, and reads shirt numbers when they become visible.

<p align="center">
  <img width="900" alt="Football Analyser output" src="https://github.com/user-attachments/assets/15e5d456-ea25-4eee-8a73-4e5deb503a5e" />
</p>

## Overview

*Technologies: Python, Ultralytics YOLO, OpenCV, BoT SORT, PyTorch, PARSeq, scikit learn and NumPy*

The analyser detects players, goalkeepers, referees and the ball, then follows them through each passage of play. It learns the two kit colours directly from the footage and uses nearby frames to keep each team assignment stable. When a shirt number is only visible briefly, the program can use that reading to label the rest of the player's track, including earlier frames.

## What It Can Do

### Keep the gameplay footage

Football broadcasts regularly cut away from the match to show the crowd, close views of players and other footage that is not useful for tracking. The first filter measures how much of each frame falls within a chosen range of green pitch colours. Frames without enough visible pitch are removed before the more expensive tracking work begins.

Once the players have been detected, a second filter looks for the main broadcast camera. A frame must contain enough visible pitch and at least eight tracked players. This removes most close views and keeps the wider views that are useful for analysing play.

Qualifying frames are joined into continuous segments for team assignment and shirt number recognition. Because the decision is based on what can be seen in the frame, a wide replay can still be included and a very tight view of live play can still be removed.

<!-- Add two examples here: one frame that was kept and one frame that was removed. -->

### Track players and assign teams

Ultralytics YOLO detects the objects in each frame. BoT SORT then gives each player a tracking ID and follows that ID through the footage. If the filters remove part of the original video, the tracker resets after the gap so that an old ID is not accidentally carried into a different passage of play.

The two teams are learned from the shirt colours in the video. The program crops the torso of each player, removes pixels that are likely to belong to the pitch, shadows or background, and groups the remaining colours using K means clustering. Blurry and heavily obstructed crops are ignored. Team readings from nearby frames are combined so that a player's colour does not keep changing when the view becomes unclear.

The finished video displays a coloured marker under each player, their tracking ID, any shirt number that was recognised, and a marker above the ball.

<p align="center">
  <img width="140" height="220" alt="Player assigned to the first team" src="https://github.com/user-attachments/assets/48f5a49a-b874-4aa1-8cd0-6d4b4baf5333" />
  &nbsp;&nbsp;&nbsp;
  <img width="140" height="220" alt="Player assigned to the second team" src="https://github.com/user-attachments/assets/ab1e24b0-c442-4d74-8fe4-962cc931470d" />
</p>

### Read shirt numbers and backtrack

Shirt numbers are often only clear for a handful of frames. The analyser samples several points from each player's track and looks for frames where the player is large enough, reasonably sharp and not heavily covered by another player. PARSeq then attempts to read the number from several areas of the shirt.

One confident reading is not enough. The same number must be found in at least two separate frames before it is accepted. The program then adds that number to the whole track, including earlier frames where the player's back was turned away or the number was too blurred to read.

In the below example, the first picture shows the correct number 6 above the players head despite it not being visible. This is because later in the footage (image 2), the number can be read and so all previous instances of that track ID now include that number recognised. 

<p align="center">
  <img width="130" height="260" alt="Earlier view of the tracked player" src="https://github.com/user-attachments/assets/1ae50868-8b69-4987-a5c5-3efdf7651aa4" />
  &nbsp;&nbsp;&nbsp;
  <img width="130" height="260" alt="Later view where the shirt number is clear" src="https://github.com/user-attachments/assets/e9286512-7492-49d2-a2d0-cae026190046" />
</p>

### Handle uncertain identities

Tracking IDs are not always reliable when players cross paths, disappear from view or become hidden behind one another. To stop one shirt number from spreading to the wrong person, the program splits a track when it finds a gap in the detections, a sudden change in position or a change of team.

It also checks for two players on the same team being given the same number at the same time. If their tracks overlap for at least ten frames, the reading with stronger support is kept and the weaker one is removed. When there is not enough evidence, the player keeps their tracking ID without being given a shirt number.
