"""Live-camera port components (see REALTIME_SAM2_OVERVIEW.md).

frame_source     -- Frame dataclass + KinectSource (Azure Kinect) + ZedSource (ZED 2)
                    + ReplaySource (chunks)
sam2_segmenter   -- SAM2 streaming mask generator with the mandatory cleanup
"""
