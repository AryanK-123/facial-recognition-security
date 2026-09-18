# Facial Recognition Security System

Hello! This is my first project and its a real-time facial recognition system built with computer vision and deep learning.
The program utilizes my webcam to detect an initial reference image and later compares it to anyone else that comes into the frame.
If anyone else besides me comes on the frame a black border and the word "UNKNOWN" is presented, otherwise my name is shown with a green outline.

## How It Works

1. **Person Detection** — [YOLO](https://github.com/ultralytics/ultralytics) This does the heavy lifting and detects people through my webcam, in the code known as YOLO
2. **Face Cropping** — The top portion of my face is cropped to compare with other frames
3. **Face Embedding** — The cropped face is passed through `facenet-pytorch` to generate an embedding which is a numerical representation such that it can be compared with other frames and could be graphed to see if any other points lie near that initial point, if so its most likely to match
4. **Comparison** — Every point is calculated through Euclidean distance (distance formula) to determine if each point is adequate or very far away
5. **Display** — On my webcam a bounding box is drawn around the frame with green and my name "Aryan" or "UNKNOWN" otherwise

## Files
- `security.py` — The main script, compiles the webcam, bounding box, and processing the frames
- `securitytrainer.py` — Generates the embeddings and compares the distances of points

## Notes
- A reference image should be used if you want to test it. This can be done through replacing my file address of the screenshot with another PNG image
- The face-cropping method isn't 100% concrete since it uses a more straightforward approach by cutting the person's face and leaving the shoulders out and minimal face adjustments can make "UNKNOWN" appear
- The similarity threshold was determined by testing the code (the 1.15 in the embedding function in `securitytrainer.py`) so it has been increased which means more leniency

## Motivation
I built this independently to explore computer vision and deep learning through a cool facial detection project to give myself a challenge
